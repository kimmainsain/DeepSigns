# measure_frr.py — b.npy 기준 FRR + PK기준 병행, BER 단위(%)→비율 정규화
import os, sys, json, math, hashlib
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset
import matplotlib.pyplot as plt

from models.mlp import MLP
from utils import (
    get_activations,
    extract_WM_from_activations,
    compute_BER,   # 주의: 이 함수가 % 단위를 반환할 수 있음
)

# ── 고정 파라미터 ────────────────────────────────────────────────
PLOTS_DIR = "plots"; os.makedirs(PLOTS_DIR, exist_ok=True)
MARK_DIR   = "logs/whitebox/mlp/marked"
MODEL_PATH = os.path.join(MARK_DIR, "mlp_nft.pth")
A_PATH     = os.path.join(MARK_DIR, "projection_matrix.npy")
PK_HEX_TXT = os.path.join(MARK_DIR, "pk_hex.txt")
PK_HASH_TXT= os.path.join(MARK_DIR, "pk_hash.txt")
B_PATH     = os.path.join(MARK_DIR, "b.npy")
B_MISS     = os.path.join(MARK_DIR, "b_mismatch.npy")

K_REPEATS      = 50
TAUS           = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]  # 비율(0~1)
CLASS_LABEL    = 0
PER_CLASS_CAP  = 2048

# ── 유틸 ─────────────────────────────────────────────────────────
def clean_hex(h: str) -> str:
    h = h.strip().lower()
    return h[2:] if h.startswith("0x") else h

def sha256_prefix_bits_from_pk_hex(pk_hex: str, t_len: int) -> np.ndarray:
    d = hashlib.sha256(bytes.fromhex(clean_hex(pk_hex))).digest()
    bits = np.unpackbits(np.frombuffer(d, np.uint8))
    return bits[:min(t_len, bits.size)].astype(np.uint8)

def bits_from_pkhash_hex(pk_hash_hex: str, t_len: int) -> np.ndarray:
    raw = bytes.fromhex(clean_hex(pk_hash_hex))
    bits = np.unpackbits(np.frombuffer(raw, np.uint8))
    return bits[:min(t_len, bits.size)].astype(np.uint8)

def load_b_saved_or_exit(T_len: int) -> np.ndarray:
    if os.path.exists(B_PATH):
        b = np.load(B_PATH)
    elif os.path.exists(B_MISS):
        b = np.load(B_MISS)
    else:
        raise SystemExit(f"[ERR] b.npy / b_mismatch.npy 가 없습니다: {B_PATH}")
    b = np.asarray(b)
    if b.ndim == 2:         # (T, C) → class 0 칼럼 사용(훈련 시 클래스별 타일링 가정)
        return b[:, 0].astype(np.uint8).reshape(-1)
    elif b.ndim == 1:
        return b.reshape(-1).astype(np.uint8)
    else:
        raise SystemExit(f"[ERR] b.npy shape 지원 안함: {b.shape}")

def class0_subset(train_ds, k: int, seed: int) -> Subset:
    if hasattr(train_ds, "targets"):
        targets = np.array(train_ds.targets, dtype=np.int64)
    else:
        targets = np.array([train_ds[i][1] for i in range(len(train_ds))], dtype=np.int64)
    idx0 = np.where(targets == CLASS_LABEL)[0]
    if idx0.size == 0:
        raise SystemExit("[ERR] MNIST(train)에서 class 0 샘플이 없습니다.")
    k = min(k, idx0.size)
    rng = np.random.RandomState(seed)
    pick = rng.choice(idx0, size=k, replace=False)
    return Subset(train_ds, pick)

def extract_once(net, A, ds_subset) -> np.ndarray:
    acts = get_activations(net, ds_subset)                 # verify.py와 동일 (device X)
    bprime = extract_WM_from_activations(acts, A)          # (T,) or (T,C)
    bp = bprime if isinstance(bprime, np.ndarray) else np.array(bprime)
    return bp.reshape(-1).astype(np.uint8)

def ber_frac(x_bits: np.ndarray, y_bits: np.ndarray) -> float:
    """utils.compute_BER가 %를 반환할 수도 있으므로 비율(0~1)로 정규화."""
    m = min(x_bits.size, y_bits.size)
    if m == 0:
        return float('nan')
    val = float(compute_BER(x_bits[:m], y_bits[:m]))
    return val/100.0 if val > 1.0 else val

def wilson(p_hat, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    denom = 1 + z**2/n
    center = (p_hat + z**2/(2*n))/denom
    margin = z*math.sqrt((p_hat*(1-p_hat)+z**2/(4*n))/n)/denom
    return (max(0.0, center - margin), min(1.0, center + margin))

# ── 파일/모델 로드 ───────────────────────────────────────────────
for p in (MODEL_PATH, A_PATH):
    if not os.path.exists(p):
        raise SystemExit(f("[ERR] 파일 없음: {p}"))

device = "cuda" if torch.cuda.is_available() else "cpu"
net = MLP().to(device)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.eval()

A = np.load(A_PATH)                 # (T, D)
T_len = int(A.shape[0])

# b 저장본(진실값) & pk기준(진단용)
b_saved = load_b_saved_or_exit(T_len)

# pk_hash.txt가 있으면 그걸 1순위로 사용(훈련 시 저장본과 정확히 일치해야 정상)
b_pk = None
if os.path.exists(PK_HASH_TXT):
    pk_hash_hex = Path(PK_HASH_TXT).read_text().strip()
    b_pk = bits_from_pkhash_hex(pk_hash_hex, T_len)
elif os.path.exists(PK_HEX_TXT):
    pk_hex = Path(PK_HEX_TXT).read_text().strip()
    b_pk = sha256_prefix_bits_from_pk_hex(pk_hex, T_len)

# 데이터셋(train)과 class0 표본 규모
tf = T.Compose([T.ToTensor()])
train_ds = torchvision.datasets.MNIST("./data", train=True, transform=tf, download=True)
if hasattr(train_ds, "targets"):
    total_c0 = int((np.array(train_ds.targets) == CLASS_LABEL).sum())
else:
    total_c0 = sum(int(train_ds[i][1] == CLASS_LABEL) for i in range(len(train_ds)))
per_iter_samples = min(PER_CLASS_CAP, total_c0)

# ── 초기 프로브: A / −A 중 b_saved 기준 BER 작은 쪽 선택 ─────────
subset0 = class0_subset(train_ds, per_iter_samples, seed=0)
bp_A  = extract_once(net,  A, subset0)
bp_nA = extract_once(net, -A, subset0)

ber_A_saved  = ber_frac(bp_A,  b_saved)
ber_nA_saved = ber_frac(bp_nA, b_saved)

if b_pk is not None:
    ber_A_pk  = ber_frac(bp_A,  b_pk)
    ber_nA_pk = ber_frac(bp_nA, b_pk)
else:
    ber_A_pk = ber_nA_pk = None

if ber_A_saved <= ber_nA_saved:
    A_use, A_choice, initial_ber_saved, initial_ber_pk =  A,  "A",  ber_A_saved,  ber_A_pk
else:
    A_use, A_choice, initial_ber_saved, initial_ber_pk = -A, "-A", ber_nA_saved, ber_nA_pk

# b_saved vs b_pk 진단
ber_saved_vs_pk = None
if b_pk is not None:
    ber_saved_vs_pk = ber_frac(b_saved, b_pk)

# ── K회 반복: b′ 재추출 → 두 기준으로 동시 집계 ───────────────────
bers_saved = np.empty(K_REPEATS, dtype=np.float32)
bers_pk    = np.empty(K_REPEATS, dtype=np.float32) if b_pk is not None else None

for i in range(K_REPEATS):
    subset_i = class0_subset(train_ds, per_iter_samples, seed=i+1)
    bp_i = extract_once(net, A_use, subset_i)
    bers_saved[i] = ber_frac(bp_i, b_saved)
    if b_pk is not None:
        bers_pk[i] = ber_frac(bp_i, b_pk)

# ── FRR(τ) 계산(비율 단위) ───────────────────────────────────────
frr_saved = {tau: float(np.mean(bers_saved > tau)) for tau in TAUS}
frr_saved_ci = {tau: wilson(frr_saved[tau], K_REPEATS) for tau in TAUS}

if b_pk is not None:
    frr_pk = {tau: float(np.mean(bers_pk > tau)) for tau in TAUS}
    frr_pk_ci = {tau: wilson(frr_pk[tau], K_REPEATS) for tau in TAUS}
else:
    frr_pk = frr_pk_ci = None

# ── 그림 저장 ────────────────────────────────────────────────────
plt.figure()
plt.hist(bers_saved, bins=30, density=True, label="BER vs b_saved")
if bers_pk is not None:
    plt.hist(bers_pk, bins=30, density=True, alpha=0.5, label="BER vs b_pk")
plt.xlabel("BER (fraction)"); plt.ylabel("Density"); plt.legend()
plt.title(f"Owner BER distributions (K={K_REPEATS}, mapping={A_choice})")
plt.savefig(os.path.join(PLOTS_DIR, "frr_hist.png"), dpi=160); plt.close()

plt.figure()
xs = TAUS
ys_saved = [frr_saved[t] for t in xs]
plt.plot(xs, ys_saved, marker="o", label="FRR vs b_saved")
if frr_pk is not None:
    ys_pk = [frr_pk[t] for t in xs]
    plt.plot(xs, ys_pk, marker="x", label="FRR vs b_pk")
plt.grid(True, which="both"); plt.legend()
plt.xlabel("τ threshold"); plt.ylabel("FRR = P[BER > τ]")
plt.title(f"FRR vs τ (K={K_REPEATS}, T={T_len}, class={CLASS_LABEL})")
plt.savefig(os.path.join(PLOTS_DIR, "frr_vs_tau.png"), dpi=160); plt.close()

# ── 메트릭 저장/출력 ─────────────────────────────────────────────
metrics = {
    "T": T_len,
    "A_shape": list(A.shape),
    "A_choice": A_choice,
    "initial_probe": {
        "ber_saved": initial_ber_saved,
        "ber_pk": initial_ber_pk
    },
    "b_saved_vs_b_pk_ber": ber_saved_vs_pk,
    "K_REPEATS": K_REPEATS,
    "per_iter_samples": per_iter_samples,
    "class_label": CLASS_LABEL,
    "FRR_saved": {str(k): v for k, v in frr_saved.items()},
    "FRR_saved_CI": {str(k): v for k, v in frr_saved_ci.items()},
    "bers_saved_mean": float(np.mean(bers_saved)),
    "bers_saved_std":  float(np.std(bers_saved, ddof=1)),
}
if frr_pk is not None:
    metrics.update({
        "FRR_pk": {str(k): v for k, v in frr_pk.items()},
        "FRR_pk_CI": {str(k): v for k, v in frr_pk_ci.items()},
        "bers_pk_mean": float(np.mean(bers_pk)),
        "bers_pk_std":  float(np.std(bers_pk, ddof=1)),
    })

with open("metrics_frr.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
