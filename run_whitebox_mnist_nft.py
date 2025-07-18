# -*- coding: utf-8 -*-
"""
run_whitebox_mnist_nft.py
────────────────────────────────────────────────────────
DeepSigns + NFT 서명 데모 (MNIST – MLP)
────────────────────────────────────────────────────────
"""
import os, sys, datetime, hashlib, json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch, torch.nn.functional as F
import torchvision
from torchvision import transforms
from dotenv import load_dotenv
import ecdsa

from models.mlp import MLP
from utils import (train_whitebox, subsample_training_data,
                   get_activations, extract_WM_from_activations, compute_BER)

# ── 프로젝트 루트 경로 삽입 (import 용) ───────────────── #
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))

# ─────────────────────────────────────────────────────── #
def run(args):
    load_dotenv()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 0‑1) ECDSA 키 로드
    sk_hex = os.getenv("PRIVATE_KEY_HEX")         # .env 키 이름 일치!
    if not sk_hex:
        raise ValueError("PRIVATE_KEY_HEX not found in .env")
    sk = ecdsa.SigningKey.from_string(bytes.fromhex(sk_hex),
                                      curve=ecdsa.SECP256k1)
    pk = sk.get_verifying_key()
    print("[OK] ECDSA keypair loaded")

    # 0‑2) MNIST 데이터
    tf = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST("./data", train=True,
                                          transform=tf, download=True)
    testset  = torchvision.datasets.MNIST("./data", train=False,
                                          transform=tf, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, 32, shuffle=False)
    testloader  = torch.utils.data.DataLoader(testset,  512, shuffle=False)

    # 1) pk_hash → 비트열 b
    pk_hash = hashlib.sha256(pk.to_string("uncompressed")).digest()
    pk_hash_hex = pk_hash.hex()
    bits256 = np.unpackbits(np.frombuffer(pk_hash, dtype=np.uint8))
    b = np.tile(bits256, (args.n_classes, 1)).T        # (256,10)
    print(f"[OK] Watermark b generated  shape={b.shape}")

    # 2) DeepSigns 학습
    model   = MLP().to(device)
    centers = torch.nn.Parameter(torch.rand(args.n_classes, 512, device=device),
                                 requires_grad=True)
    opt = torch.optim.RMSprop([{'params': model.parameters()},
                               {'params': centers}], lr=args.lr)
    print("[INFO] embedding watermark …")
    train_whitebox(model, opt, trainloader, b, centers, args,
                   save_path='logs/whitebox/mlp/marked/projection_matrix.npy')
    print("[OK] embedding done")

    # 3) 모델 저장 & 성능 확인
    model.eval()
    loss_sum = acc_sum = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
            acc_sum  += logits.argmax(1).eq(y).sum().item()
    print(f"Test Loss={loss_sum/len(testloader.dataset):.4f} "
          f"Acc={acc_sum/len(testloader.dataset):.4f}")

    sd_path = 'logs/whitebox/mlp/marked/mlp_nft.pth'
    os.makedirs(os.path.dirname(sd_path), exist_ok=True)
    torch.save(model.state_dict(), sd_path)

    # 4) metadata.json (msg, sig, pk_hash)
    ts_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    message = f"Model=MLP_MNIST;Timestamp={ts_iso};pk_hash={pk_hash_hex}"
    signature = sk.sign_deterministic(message.encode(),
                                      hashfunc=hashlib.sha256)
    meta = {"msg": message, "sig": signature.hex(), "pk_hash": pk_hash_hex}
    meta_path = 'logs/whitebox/mlp/marked/metadata.json'
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print("[OK] metadata.json saved")

    # 5) 워터마크 추출 & 검증
    marked_model = MLP().to(device)
    marked_model.load_state_dict(torch.load(sd_path))
    A = np.load('logs/whitebox/mlp/marked/projection_matrix.npy')

    subset = subsample_training_data(trainset, args.target_class)
    activs = get_activations(marked_model, subset)
    bits_dec = extract_WM_from_activations(activs, A)
    ber = compute_BER(bits_dec, b[:, args.target_class])
    print(f"BER(class {args.target_class}) = {ber:.4f}")

    if ber <= 0.01:
        decoded_hex = np.packbits(bits_dec).tobytes().hex()
        print("Hash recovered:", decoded_hex[:32], "…")
        print("Match:", decoded_hex == pk_hash_hex)
    else:
        print("BER too high → hash comparison skipped")

    # 서명 검증
    ok = pk.verify(bytes.fromhex(meta["sig"]), meta["msg"].encode(),
                   hashfunc=hashlib.sha256)
    print("Signature valid:", ok)

# ── CLI ─────────────────────────────────────────────── #
def main():
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('--n_classes', type=int, default=10)
    p.add_argument('--lr',        type=float, default=0.001)
    p.add_argument('--epochs',    type=int,   default=30)
    p.add_argument('--scale',     type=float, default=0.05)
    p.add_argument('--gamma',     type=float, default=0.05)
    p.add_argument('--embed_bits',type=int,   default=256)
    p.add_argument('--target_class', type=int, default=0)
    args = p.parse_args()
    args.embed_bits = 256         # 해시 길이 고정
    run(args)

if __name__ == '__main__':
    main()
