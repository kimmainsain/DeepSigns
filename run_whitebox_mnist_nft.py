# -*- coding: utf-8 -*-
"""
run_whitebox_mnist.py  (훈련 전용)
────────────────────────────────────────────────────────
산출물 (logs/whitebox/mlp/marked/):
  • mlp_nft.pth           : 워터마크 삽입 모델
  • projection_matrix.npy : DeepSigns 투영 행렬 A
  • pk_hex.txt            : 언컴프레스트 공개키
  • pk_hash.txt           : SHA‑256 해시 (256bit)
────────────────────────────────────────────────────────
"""
import os, sys, hashlib, random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import random, numpy as np, torch
import numpy as np, torch, torchvision
import torch.nn.functional as F                     # ← 추가
from torchvision import transforms
from dotenv import load_dotenv
import ecdsa

from models.mlp import MLP
from utils import train_whitebox, make_balanced_loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run(cfg):
    set_seed(0)
    load_dotenv()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 개인키 → 공개키
    sk_hex = os.getenv("PRIVATE_KEY_HEX")
    if not sk_hex:
        raise ValueError("PRIVATE_KEY_HEX missing in .env")
    sk = ecdsa.SigningKey.from_string(bytes.fromhex(sk_hex), curve=ecdsa.SECP256k1)
    pk_bytes = sk.get_verifying_key().to_string("uncompressed")

    # 공개키 해시 → 워터마크 비트
    pk_hash = hashlib.sha256(pk_bytes).digest()
    bits256 = np.unpackbits(np.frombuffer(pk_hash, dtype=np.uint8))
    b       = np.tile(bits256, (cfg.n_classes, 1)).T   # (256×10)
    # # ─── (NEW) b.npy 즉시 저장 ───────────────────────────
    # out_dir = "logs/whitebox/mlp/marked"
    # os.makedirs(out_dir, exist_ok=True)
    # np.save(f"{out_dir}/b.npy", b.astype(np.uint8))
    # print(f"[INFO] original b saved → {out_dir}/b.npy  (shape {b.shape})")

    # 데이터 로드
    tf = transforms.Compose([transforms.ToTensor()])
    tr = torchvision.datasets.MNIST("./data", train=True,  transform=tf, download=True)
    ts = torchvision.datasets.MNIST("./data", train=False, transform=tf, download=True)
    tr_loader = make_balanced_loader(tr, cfg.target_class, batch_size=32)
    ts_loader = torch.utils.data.DataLoader(ts, 512, shuffle=False)   # ← 이름 고정

    # DeepSigns 학습
    model = MLP().to(dev)
    centers = torch.nn.Parameter(torch.rand(cfg.n_classes, 512, device=dev), requires_grad=True)
    optim = torch.optim.RMSprop(
        [{"params": model.parameters()}, {"params": centers}], lr=cfg.lr)
    train_whitebox(model, optim, tr_loader, b, centers, cfg,
                    save_path="logs/whitebox/mlp/marked/projection_matrix.npy")

    # 3) 테스트 손실·정확도
    model.eval()
    loss_meter = 0.0
    acc_meter  = 0
    with torch.no_grad():
        for data, target in ts_loader:
            data, target = data.to(dev), target.to(dev)
            logits, _ = model(data)
            loss_meter += F.cross_entropy(logits, target, reduction='sum').item()
            acc_meter  += logits.argmax(1).eq(target).sum().item()

    num_samples = len(ts_loader.dataset) 
    print('Test loss    :', loss_meter / num_samples)
    print('Test accuracy:', acc_meter  / num_samples)

    # 산출물 저장
    out = "logs/whitebox/mlp/marked"
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), f"{out}/mlp_nft.pth")
    with open(f"{out}/pk_hex.txt",  "w") as f: f.write("04" + pk_bytes.hex())
    with open(f"{out}/pk_hash.txt", "w") as f: f.write(pk_hash.hex())
    print("[OK] model / pk_hex / pk_hash saved")

def main():
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--lr",        type=float, default=0.001)
    p.add_argument("--epochs",    type=int,   default=30)
    p.add_argument("--scale",     type=float, default=0.02)
    p.add_argument("--gamma",     type=float, default=0.15)
    p.add_argument('--target_class', type=int, default=0)

    args = p.parse_args()
    args.embed_bits = 256
    run(args)

if __name__ == "__main__":
    main()
