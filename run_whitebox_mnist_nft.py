# -*- coding: utf-8 -*-
"""
run_whitebox_mnist.py  (훈련 전용)
────────────────────────────────────────────────────────
산출물 (logs/whitebox/mlp/marked/):
  • mlp_nft.pth           : 워터마크 삽입 모델
  • projection_matrix.npy : DeepSigns 투영 행렬 A   (T=16 기준)
  • pk_hex.txt            : 언컴프레스트 공개키
  • pk_hash.txt           : SHA-256 해시 (256bit 전체)
  • b.npy                 : 사용한 16비트 b (선택 저장)
────────────────────────────────────────────────────────
"""
import os, sys, hashlib, random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np, torch, torchvision
import torch.nn.functional as F
from torchvision import transforms
from dotenv import load_dotenv
import ecdsa

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
    # 1. (Fig. 1 ③-①)  워터마크 비트열 b 생성 (PK 해시 앞 32bit)
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

    # 앞쪽 embed_bits 만큼 비트 추출 (기본 32)
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
    with open(os.path.join(out_dir, 'pk_hex.txt'), 'w') as f:
        f.write("04" + pk_bytes.hex())
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
    with open(f"{out_dir}/pk_hex.txt",  "w") as f: f.write("04" + pk_bytes.hex())
    with open(f"{out_dir}/pk_hash.txt", "w") as f: f.write(pk_hash.hex())
    print("[OK] model / pk_hex / pk_hash / b.npy saved")
    
    
        # ========== (1) A → A.npy로 저장 ==========
    A_src = "logs/whitebox/mlp/marked/projection_matrix.npy"
    A = np.load(A_src).astype(np.float32)

    zkvm_data_dir = os.path.expanduser("~/zkvm/wm_proof/data")
    os.makedirs(zkvm_data_dir, exist_ok=True)
    np.save(os.path.join(zkvm_data_dir, "A.npy"), A)

    # ========== (2) mu 계산 후 mu.npy 저장 ==========
    #  - 여기선 'target_class'의 샘플들로 MLP의 은닉표현 평균을 뮤로 사용
    #  - 모델은 위에서 학습된 'model' (model(x) -> (logits, feat) 가정)

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

    # 학습에 썼던 'tr'(train MNIST), 'args.target_class', 'dev' 그대로 활용
    mu_vec = compute_mu(model, trainset, args.target_class, device, batch_size=512)
    np.save(os.path.join(zkvm_data_dir, "mu.npy"), mu_vec)

    # (선택) 공개키도 같이 복사
    with open(os.path.join(zkvm_data_dir, "pk_hex.txt"), "w") as f:
        f.write(pk_bytes.hex())

    print(f"[OK] Saved for zkVM: {zkvm_data_dir}/A.npy, {zkvm_data_dir}/mu.npy, pk_hex.txt")


def main():
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--n_classes",     type=int,   default=10)
    p.add_argument("--lr",            type=float, default=0.001)
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--scale",         type=float, default=0.02)
    p.add_argument("--gamma",         type=float, default=0.15)
    p.add_argument("--target_class",  type=int,   default=0)
    p.add_argument('--embed_bits', type=int, default=32,
                        help='N : number of watermark bits per class (≤256)')
    args = p.parse_args()

    run(args)

if __name__ == "__main__":
    main()
