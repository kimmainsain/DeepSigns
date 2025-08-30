# -*- coding: utf-8 -*-
"""
run_whitebox_mnist.py
───────────────────────────────────────────────────────────────────────────────
DeepSigns – MNIST 화이트박스(N-bit) 워터마크 **삽입 + 추출** 실험 스크립트

● 논문 대응 흐름
    Fig. 1 ③     :   b(비트열) 생성 ➊  +  MLP + centers(μ) 초기화 ➋
                     train_whitebox() 호출 → 식 (1)(3) 학습 ➌
    테스트        :   원본 정확도 확인
    Alg. 3       :   키 서브셋 → μ′ → A·μ′ → BER 계산 ➍
───────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from models.mlp import MLP
from utils import *
import hashlib

from dotenv import load_dotenv
import ecdsa

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

    # 모델 파라미터(워터마크 포함) 저장
    sd_path = os.path.join(out_dir, 'mlp.pth')
    torch.save(model.state_dict(), sd_path)

    # ───────────────────────────────────────────────────────── #
    # 5. 워터마크 추출·검증 (Alg. 3)
    # ───────────────────────────────────────────────────────── #
    marked_model = MLP().to(device)
    marked_model.load_state_dict(torch.load(sd_path, map_location=device))

    subset_loader = subsample_training_data(trainset, args.target_class)
    activations = get_activations(marked_model, subset_loader)
    print("Collected activations of first WM-carrying dense layer")

    A = np.load(os.path.join(out_dir, 'projection_matrix.npy'))
    decoded_bits = extract_WM_from_activations(activations, A)

    BER = compute_BER(decoded_bits, b[:, args.target_class])
    print(f"BER for class {args.target_class} = {BER}")

    print("A shape :", A.shape)
    print("first 3 rows of A\n", A[:3])
    print("b shape :", b.shape)
    print("b[:, 0] =", b[:, 0])   # 예: 클래스 0 워터마크

    np.savetxt('A_matrix.txt', A, fmt='%.6f', delimiter=',')
    np.savetxt('b_bits.txt', b, fmt='%d', delimiter='')

    def sha256_of_npy(path):
        data = np.load(path)
        return hashlib.sha256(data.tobytes()).hexdigest()

    print("SHA256(A) :", sha256_of_npy(os.path.join(out_dir, 'projection_matrix.npy')))
    print("SHA256(b) :", sha256_of_npy(os.path.join(out_dir, 'b.npy')))
    counts = np.bincount(np.array(trainset.targets))
    print(counts, counts.sum())  # 클래스별 개수와 총합(60000)


# ────────────────────────────────────────────────────────────── #
#  CLI 인수 정의
# ────────────────────────────────────────────────────────────── #
def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # 일반 파라미터
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes in MNIST (10 digits)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for WM training')

    # 워터마킹 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=1,
                        help='embed_epoch : train_whitebox epochs')
    parser.add_argument('--scale', type=float, default=0.01,
                        help='λ₁ (loss1+2+3 weight)')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='λ₂ (watermark CE weight)')
    parser.add_argument('--target_dense_idx', type=int, default=2,
                        help='(unused for this MLP) Dense layer index placeholder')

    # ★ 기본값 32로 변경: SHA-256 앞 32bit 사용
    parser.add_argument('--embed_bits', type=int, default=32,
                        help='N : number of watermark bits per class (≤256)')

    parser.add_argument('--target_class', type=int, default=0,
                        help='Digit (0-9) chosen for extraction demo')
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
