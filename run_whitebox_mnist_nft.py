# -*- coding: utf-8 -*-
"""
run_whitebox_mnist_nft.py
───────────────────────────────────────────────────────────────────────────────
DeepSigns + NFT 서명 데모 (MNIST - MLP)

● 전체 흐름
    (1) .env 파일의 개인키(sk) 로드 → 공개키(pk), pk_hash(256bit) 산출
    (2) pk_hash 비트열을 DeepSigns 워터마크(b) 로 사용해 모델 학습
    (3) 학습 완료 후  NFT 메타데이터:
            M = "ModelID + ISO8601 시각 + pk_hash"
            Sig = Sign_SHA256(sk, M)
    (4) xkey 로 중간층 평균 μ′ 획득 → b′ 복원(=pk_hash′) → BER 계산
    (5) BER ≤ τ  AND  pk_hash′ == pk_hash  AND  Verify_SHA256(pk, M, Sig)
        ⇒ 모델 - NFT 소유권 일치 증명
───────────────────────────────────────────────────────────────────────────────
"""

import os, sys, datetime, hashlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch, torch.nn.functional as F
import torchvision
from torchvision import transforms
from dotenv import load_dotenv
import ecdsa                              # ECDSA(secp256k1) 라이브러리

from models.mlp import MLP
from utils import *                       # DeepSigns util (train_whitebox 등)

# ── 프로젝트 루트 경로 삽입 ──────────────────────────────────────────────── #
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))


def run(args):
    # 0. 환경·키·데이터 준비 -------------------------------------------------- #
    load_dotenv()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # (a) ECDSA 키 불러오기 (개인키 HEX → SigningKey)
    sk_hex = os.getenv("PRIVATE_KEY_HEX")
    if not sk_hex:
        raise ValueError(
            "PRIVATE_KEY_HEX not found in .env. "
            "예) PRIVATE_KEY_HEX=1c0f... (64 hex chars)")
    sk = ecdsa.SigningKey.from_string(
        bytes.fromhex(sk_hex), curve=ecdsa.SECP256k1)
    pk = sk.get_verifying_key()           # 공개키 객체
    print("[SUCCESS] ECDSA key pair loaded from .env file.")

    # (b) MNIST 데이터
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root="./data/", train=True,
                                          transform=transform, download=True)
    testset  = torchvision.datasets.MNIST(root="./data/", train=False,
                                          transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=512, shuffle=False)

    # 1. 공개키 → SHA-256 해시 → 비트열 b (256 × 10)
    # ---------------------------------------------------------------------- #
    pk_bytes = pk.to_string("uncompressed")       # 0x04 + x32 + y32  = 65 B
    pk_hash  = hashlib.sha256(pk_bytes).digest()  # 32 B = 256 bit
    pk_hash_hex = pk_hash.hex()

    # 해시를 bit 배열로 변환 (np.unpackbits → uint8 array 길이 256)
    bits_256 = np.unpackbits(np.frombuffer(pk_hash, dtype=np.uint8))
    # DeepSigns 형식 (N, C) = (256, 10) 로 복제
    b = np.tile(bits_256, (args.n_classes, 1)).T
    print(f"[SUCCESS] Watermark bit string b generated. (shape={b.shape})")
    print(f"    > Public Key SHA-256 : {pk_hash_hex}")

    # 2. 모델·센터 초기화 & 워터마크 삽입 학습 ------------------------------- #
    model = MLP().to(device)                       # MLP: (logits, feat512)
    centers = torch.nn.Parameter(
        torch.rand(args.n_classes, 512, device=device), requires_grad=True)
    optimizer = torch.optim.RMSprop(
        [{'params': model.parameters()}, {'params': centers}], lr=args.lr)

    print("\n[INFO] Starting watermark embedding into the model...")
    train_whitebox(model, optimizer, trainloader,
                   b=b, centers=centers, args=args,
                   save_path='logs/whitebox/mlp/marked/projection_matrix.npy')
    print("[SUCCESS] Watermark embedding completed.")

    # 3. 모델 정확도 확인 ----------------------------------------------------- #
    model.eval()
    loss_meter = acc_meter = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss_meter += F.cross_entropy(logits, y, reduction='sum').item()
            acc_meter  += logits.argmax(1).eq(y).sum().item()
    print(f"    Test Loss    : {loss_meter / len(testloader.dataset):.4f}")
    print(f"    Test Accuracy: {acc_meter / len(testloader.dataset):.4f}")

    sd_path = 'logs/whitebox/mlp/marked/mlp_nft.pth'
    torch.save(model.state_dict(), sd_path)
    print(f"[SUCCESS] Watermarked model saved: {sd_path}")

    # 4. NFT 메타데이터(메시지+서명) 생성 ----------------------------------- #
    ts_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    message = f"Model=MLP_MNIST; Timestamp={ts_iso}; pk_hash={pk_hash_hex}"
    # SHA-256 deterministic 서명
    signature = sk.sign_deterministic(message.encode(),
                                      hashfunc=hashlib.sha256)
    print("\n[NFT Metadata]")
    print("    Message   :", message)
    print("    Signature     :", signature.hex()[:64] + "…")
    print("    Public Key Hash:", pk_hash_hex)

    # 5. 워터마크 추출 & 검증 ----------------------------------------------- #
    print("\n[안내] 워터마크를 추출합니다…")
    marked_model = MLP().to(device)
    marked_model.load_state_dict(torch.load(sd_path))
    A = np.load('logs/whitebox/mlp/marked/projection_matrix.npy')

    # target_class 데이터 절반으로 μ′ 추정
    subset = subsample_training_data(trainset, args.target_class)
    activs = get_activations(marked_model, subset)          # (m,512)
    bits_decoded = extract_WM_from_activations(activs, A)   # (256,1)

    BER = compute_BER(bits_decoded, b[:, args.target_class])
    print(f"    BER (class {args.target_class}) : {BER:.4f}")

    # 해시 복원 비교 (BER가 충분히 낮을 때만 진행)
    if BER <= 0.01:
        hash_decoded_hex = np.packbits(bits_decoded).tobytes().hex()
        print(f"    Decoded Hash:", hash_decoded_hex)
        if hash_decoded_hex == pk_hash_hex:
            print("    [SUCCESS] Hash matched (success)")
        else:
            print("    [FAILURE] Hash mismatch (failure)")
    else:
        print("    BER is too high, skipping hash comparison.")

    # 서명 검증 (공개키 + 메시지 + 서명) ------------------------------------- #
    print("\n[INFO] Verifying signature...")
    try:
        if pk.verify(signature, message.encode(), hashfunc=hashlib.sha256):
            print("    [SUCCESS] Signature valid (success) - Ownership proven!")
        else:
            print("    [FAILURE] Signature invalid (failure)")
    except ecdsa.BadSignatureError:
        print("    [실패] BadSignatureError (실패)")


# ── CLI 인자 --------------------------------------------------------------- #
def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_classes', type=int, default=10,
                        help='MNIST 클래스 수 (고정 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='RMSprop 학습률')
    parser.add_argument('--epochs', type=int, default=1,      # ← 기본 5 epoch
                        help='워터마크 학습 epoch 수')
    parser.add_argument('--scale', type=float, default=0.05,   # ← λ₁ 권장값
                        help='λ₁ : loss1+2+3 가중치')
    parser.add_argument('--gamma', type=float, default=0.05,   # ← λ₂ 권장값
                        help='λ₂ : watermark BCE 가중치')
    parser.add_argument('--embed_bits', type=int, default=256,
                        help='워터마크 bit 수 (SHA-256 해시 길이)')
    parser.add_argument('--target_class', type=int, default=0,
                        help='추출 테스트용 클래스 (0~9)')
    args = parser.parse_args()

    if args.embed_bits != 256:
        print("embed_bits is fixed to 256 (PK hash length), forcing to 256.")
        args.embed_bits = 256

    run(args)


if __name__ == '__main__':
    main()
