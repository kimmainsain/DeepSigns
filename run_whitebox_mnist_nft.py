# -*- coding: utf-8 -*-
"""
run_whitebox_mnist.py  (훈련 전용)
────────────────────────────────────────────────────────
산출물 (logs/whitebox/mlp/marked/):
  • mlp_nft.pth            : 워터마크 삽입 모델
  • projection_matrix.npy  : DeepSigns 투영 행렬 A   (T=16 기준)
  • sig_msg_hex.txt        : (기존 pk_hex.txt → 변경) 공개 아티팩트용 값
  • pk_hash.txt            : SHA-256 해시 (256bit 전체, b 생성에 사용)
  • b.npy                  : 사용한 16비트 b (선택 저장)

zkVM 입력 (zkvm/data/):
  • A.npy, mu.npy
  • A_int.bin, mu_int.bin  : i64 little-endian
  • public.json            : {h_a, h_mu, sig_msg_hex, l, tau, scale, sign_zero_rule}
────────────────────────────────────────────────────────
"""
import os, sys, hashlib, random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np, torch, torchvision
import torch.nn.functional as F
from torchvision import transforms
from dotenv import load_dotenv
import ecdsa
import json, math

from models.mlp import MLP
from utils import train_whitebox, make_balanced_loader

# ── 프로젝트 루트 경로 추가 ───────────────────────────────── #
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))

def run(args):
    # ───────────────────────────────────────────────────────── #
    # 0. 환경 설정 + MNIST 데이터 로드
    # ───────────────────────────────────────────────────────── #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(
        root="./data/", transform=transform, train=True, download=True)
    testset = torchvision.datasets.MNIST(
        root="./data/", transform=transform, train=False, download=True)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=512, shuffle=False)

    # 로그 디렉터리 보장
    out_dir = 'logs/whitebox/mlp/marked'
    os.makedirs(out_dir, exist_ok=True)

    # ───────────────────────────────────────────────────────── #
    # 1. (Fig. 1 ③-①)  워터마크 비트열 b 생성 (PK 해시 앞 T비트)
    #    b ∈ {0,1}^{embed_bits × 10}
    # ───────────────────────────────────────────────────────── #
    load_dotenv()  # .env 에 PRIVATE_KEY_HEX 필요
    sk_hex = os.getenv("PRIVATE_KEY_HEX")
    if not sk_hex:
        sys.exit("[ERR] .env 에 PRIVATE_KEY_HEX 가 없습니다")

    # 개인키 → 공개키(언컴프레스트) → SHA-256
    sk = ecdsa.SigningKey.from_string(bytes.fromhex(sk_hex), curve=ecdsa.SECP256k1)
    pk_bytes = sk.get_verifying_key().to_string("uncompressed")  # 65 bytes
    pk_hash  = hashlib.sha256(pk_bytes).digest()                  # 32 bytes (256bit)

    # 앞쪽 embed_bits 만큼 비트 추출 (default : 128 bits)
    T = int(args.embed_bits)
    if T > 256:
        sys.exit("[ERR] embed_bits가 256을 초과했습니다. (현재 구현은 SHA-256 앞부분만 사용)")

    all_bits = np.unpackbits(np.frombuffer(pk_hash, dtype=np.uint8))  # (256,)
    bitsT = all_bits[:T]                                              # (T,)
    b = np.tile(bitsT[:, None], (1, args.n_classes)).astype(np.uint8) # (T, n_classes)

    # 디버그 출력 (hex, bitstring)
    need_bytes = (T + 7) // 8  # T=32 → 4 bytes
    print(f"[INFO] pk_hash (full hex)   : {pk_hash.hex()}")
    print(f"[INFO] pk_hash first {T}bits: {pk_hash[:need_bytes].hex()} (hex, truncated to {T} bits)")
    print(f"[INFO] b shape              : {b.shape}")
    print(f"[INFO] first-{T} bitstring  : {''.join(map(str, bitsT.tolist()))}")

    # 아티팩트 저장
    np.save(os.path.join(out_dir, 'b.npy'), b)
    # (이전 pk_hex.txt → sig_msg_hex.txt 로 변경) — 현재는 공개키 바이트를 그대로 저장
    with open(os.path.join(out_dir, 'sig_msg_hex.txt'), 'w') as f:
        f.write(pk_bytes.hex())
    with open(os.path.join(out_dir, 'pk_hash.txt'), 'w') as f:
        f.write(pk_hash.hex())

    # ───────────────────────────────────────────────────────── #
    # 2. (Fig. 1 ③-②)  모델 & 센터 μ 초기화
    # ───────────────────────────────────────────────────────── #
    model = MLP().to(device)                        # forward → (logits, feat)
    centers = torch.nn.Parameter(
        torch.rand(args.n_classes, 512, device=device), requires_grad=True)

    optimizer = torch.optim.RMSprop(
        [{'params': model.parameters()},
         {'params': centers}],
        lr=args.lr, alpha=0.9, eps=1e-8, weight_decay=1e-3)

    # ───────────────────────────────────────────────────────── #
    # 3. 워터마크 삽입 학습 (투영행렬 A 저장)
    # ───────────────────────────────────────────────────────── #
    train_whitebox(model, optimizer, trainloader,
                   b=b,
                   centers=centers,
                   args=args,
                   save_path=os.path.join(out_dir, 'projection_matrix.npy'))
    
    # ───────────────────────────────────────────────────────── #
    # 4. 모델 성능 확인
    # ───────────────────────────────────────────────────────── #
    model.eval()
    loss_meter, acc_meter = 0, 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            logits, _ = model(data)
            loss_meter += F.cross_entropy(logits, target, reduction='sum').item()
            acc_meter += logits.argmax(1).eq(target).sum().item()
    print('Test loss    :', loss_meter / len(testloader.dataset))
    print('Test accuracy:', acc_meter  / len(testloader.dataset))

    # 6) 산출물 저장
    torch.save(model.state_dict(), f"{out_dir}/mlp_nft.pth")
    with open(f"{out_dir}/sig_msg_hex.txt",  "w") as f: f.write(pk_bytes.hex())
    with open(f"{out_dir}/pk_hash.txt", "w") as f: f.write(pk_hash.hex())
    print("[OK] model / sig_msg_hex / pk_hash / b.npy saved")
    
    deepsigns_root = current_dir
    zkvm_data_dir = os.path.join(deepsigns_root, 'zkvm', 'data')
    os.makedirs(zkvm_data_dir, exist_ok=True)

    # ========== (1) A → A.npy 저장 ==========
    A_src = os.path.join(out_dir, 'projection_matrix.npy')  # logs/.../projection_matrix.npy
    A = np.load(A_src).astype(np.float32)
    np.save(os.path.join(zkvm_data_dir, "A.npy"), A)

    # ========== (2) μ 계산 후 mu.npy 저장 ==========
    #  - target_class 샘플들로 MLP 은닉표현 평균을 μ로 사용
    def compute_mu(model, dataset, target_class, device, batch_size=256):
        loader = make_balanced_loader(dataset, target_class, batch_size=batch_size)
        model.eval()
        feats = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                _, f = model(x)           # f: (B, D)
                feats.append(f.detach().cpu())
        mu = torch.cat(feats, dim=0).mean(dim=0).numpy().astype(np.float32)  # (D,)
        return mu

    mu_vec = compute_mu(model, trainset, args.target_class, device, batch_size=512)
    np.save(os.path.join(zkvm_data_dir, "mu.npy"), mu_vec)

    # (선택) 공개 아티팩트도 같이 저장 — zkVM 쪽에서 정규화하므로 여기서는 raw hex로 저장
    with open(os.path.join(zkvm_data_dir, "sig_msg_hex.txt"), "w") as f:
        f.write(pk_bytes.hex())

    print(f"[OK] Saved for zkVM: {zkvm_data_dir}/A.npy, {zkvm_data_dir}/mu.npy, sig_msg_hex.txt")

    # ========== (3) A_int.bin / mu_int.bin & public.json 생성 ==========
    # - A, μ 정수화(i64, little-endian) → h_a / h_mu 해시 → public.json 기록
    try:
        # (a) 정수화 스케일/규칙
        scale = 4096                      # 고정소수 스케일 (필요시 조정)
        sign_zero_rule = "ge_zero_is_one" # z_i >= 0이면 1, 아니면 0

        # (b) 실수 → 정수화
        #    안전장치: 차원 일치 확인 (A.shape = (L, D), mu.shape = (D,))
        L, D = A.shape
        assert mu_vec.shape[0] == D, f"[ERR] mu dimension {mu_vec.shape[0]} != A.D {D}"

        A_int  = np.rint(A * scale).astype(np.int64)       # (L, D)
        mu_int = np.rint(mu_vec * scale).astype(np.int64)  # (D,)

        # (c) .bin (i64, little-endian)로 저장
        A_bin_path  = os.path.join(zkvm_data_dir, "A_int.bin")
        mu_bin_path = os.path.join(zkvm_data_dir, "mu_int.bin")
        with open(A_bin_path, "wb") as f:
            f.write(A_int.astype("<i8", copy=False).tobytes())
        with open(mu_bin_path, "wb") as f:
            f.write(mu_int.astype("<i8", copy=False).tobytes())

        # (d) h_a / h_mu = SHA256(각 .bin 바이트)
        with open(A_bin_path, "rb") as f:
            h_a = hashlib.sha256(f.read()).hexdigest()
        with open(mu_bin_path, "rb") as f:
            h_mu = hashlib.sha256(f.read()).hexdigest()

        # (e) public.json 필드 계산
        #     기본 τ = ceil(0.2 * L) (실험에 맞춰 조정 가능)
        tau = int(math.ceil(0.2 * L))
        tau = max(1, tau)

        with open(os.path.join(zkvm_data_dir, "sig_msg_hex.txt"), "r") as f:
            sig_msg_hex_for_public = f.read().strip()

        public = {
            "h_a": h_a,
            "h_mu": h_mu,                 # (신규) mu 커밋 추가
            "sig_msg_hex": sig_msg_hex_for_public,  # (pk_hex → sig_msg_hex)
            "l": int(L),
            "tau": int(tau),
            "scale": int(scale),          # 메타 정보(연산엔 영향 X)
            "sign_zero_rule": sign_zero_rule
        }

        public_path = os.path.join(zkvm_data_dir, "public.json")
        with open(public_path, "w", encoding="utf-8") as f:
            json.dump(public, f, ensure_ascii=False, indent=2)

        print(f"[OK] Generated zkVM inputs: {A_bin_path}, {mu_bin_path}, {public_path}")
        print(f"[INFO] L={L}, D={D}, tau={tau}, scale={scale}, h_a={h_a[:16]}..., h_mu={h_mu[:16]}...")

    except Exception as e:
        print(f"[WARN] Failed to generate zkVM .bin/public.json: {e}")


def main():
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--n_classes",     type=int,   default=10)
    p.add_argument("--lr",            type=float, default=0.001)
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--scale",         type=float, default=0.02)
    p.add_argument("--gamma",         type=float, default=0.15)
    p.add_argument("--target_class",  type=int,   default=0)
    p.add_argument('--embed_bits', type=int, default=128,
                        help='N : number of watermark bits per class (≤256)')
    args = p.parse_args()

    run(args)

if __name__ == "__main__":
    main()
