# measure_far.py
import os, json, math, numpy as np, torch, ecdsa, hashlib, matplotlib.pyplot as plt
from common_wm import bits_from_pk, ber, extract_bprime_unique
import pathlib

os.makedirs("plots", exist_ok=True)  # savefig 호출 전에 디렉터리 생성
# ==== 설정 ====
T = 128               # 유효 비트 길이
N_ATTACK = 1_000_000    # 공격자 키 표본 수 (실무 1e6까지 증대 가능)
TAUS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.2]

# ---- 오너 모델 로드 ----
from models.mlp import MLP
A = np.load("logs/whitebox/mlp/marked/projection_matrix.npy")   # (T,D)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP()
model.load_state_dict(torch.load("logs/whitebox/mlp/marked/mlp_nft.pth",
                                 map_location=device))
bprime = extract_bprime_unique(model, A, device=device)  # (T,)

# ---- 공격자 키 생성 & BER 분포 ----
# def gen_random_pk_uncompressed() -> bytes:
#     sk = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
#     vk = sk.get_verifying_key()
#     return vk.to_string("uncompressed")  # 65 bytes, 0x04 || X || Y

# 빠르게 돌리는 임시테스트
def gen_random_pk_uncompressed():
    import os
    return b"\x04" + os.urandom(64)

bers = np.empty(N_ATTACK, dtype=np.float32)
for i in range(N_ATTACK):
    pk_u = gen_random_pk_uncompressed()
    b_star = bits_from_pk(pk_u, T)       # (T,)
    bers[i] = ber(b_star, bprime)

# ---- FAR 추정 & 이론치 ----
def FAR_theory(L, tau):
    m = int(math.floor(tau*L))
    s=0
    for k in range(m+1):
        s += math.comb(L,k)
    return s/(2**L)

far_emp = {tau: float(np.mean(bers <= tau)) for tau in TAUS}
far_the = {tau: FAR_theory(T, tau) for tau in TAUS}

# ---- 신뢰구간(윌슨) ----
def wilson(p_hat, n, z=1.96):
    if n==0: return (0,0)
    denom = 1 + z**2/n
    center = (p_hat + z**2/(2*n))/denom
    margin = z*math.sqrt((p_hat*(1-p_hat)+z**2/(4*n))/n)/denom
    return (max(0.0, center-margin), min(1.0, center+margin))

far_ci = {tau: wilson(far_emp[tau], N_ATTACK) for tau in TAUS}

# ---- 히스토그램 + 이론 PDF ----
plt.figure()
plt.hist(bers, bins=50, density=True, label="empirical")
# 이론 정규 근사 (N(0.5, 0.25/T))
mu, sigma = 0.5, (0.25 / T) ** 0.5
xs_pdf = np.linspace(0, 1, 400)
pdf = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*((xs_pdf-mu)/sigma)**2)
plt.plot(xs_pdf, pdf, linestyle="--", label=f"Normal approx (μ=0.5, σ≈{sigma:.4f})")

# τ 수직선
for t in TAUS:
    plt.axvline(t, linestyle=":", alpha=0.6)

plt.xlim(0, 1)
plt.xlabel("BER(b*, b')")
plt.ylabel("Density")
plt.legend()
plt.title(f"Attacker BER distribution (N={N_ATTACK}, T={T})")
plt.savefig("plots/far_hist.png", dpi=160); plt.close()

# ---- FAR(τ): 이론 vs 경험(95% 상한) ----
# 기대 히트수도 계산해 주면 왜 0인지 설득력 ↑
expected_hits = {tau: N_ATTACK * far_the[tau] for tau in TAUS}

xs = TAUS
ys_the = [far_the[t] for t in xs]
ys_upper = [far_ci[t][1] for t in xs]   # 윌슨 95% 상한

plt.figure()
plt.plot(xs, ys_the, marker="x", label="theory (binom tail)")
plt.plot(xs, ys_upper, marker="o", linestyle="--",
         label="empirical (95% upper bound)")
plt.yscale("log")
plt.grid(True, which="both")
plt.xlabel("τ threshold")
plt.ylabel("FAR (or upper bound)")
plt.legend()
plt.title(f"FAR vs τ (T={T}, N={N_ATTACK})")
plt.savefig("plots/far_vs_tau.png", dpi=160); plt.close()

# (선택) 텍스트 로그에도 기대 히트수 추가
print("expected_hits:", {str(t): f"{expected_hits[t]:.3e}" for t in TAUS})

# ---- 저장 ----
metrics = {
  "T": T, "N_ATTACK": N_ATTACK,
  "far_emp": far_emp, "far_theory": far_the, "far_ci": {str(k): v for k,v in far_ci.items()},
  "bers_mean": float(np.mean(bers)), "bers_std": float(np.std(bers, ddof=1))
}
os.makedirs("plots", exist_ok=True)
with open("metrics.json", "w") as f: json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
