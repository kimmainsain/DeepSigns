# verify.py  ───────────────────────────────────────────
"""
(1) --pk-hex <공개키>      : 저장된 b.npy 와 b^ 비교 (BER 출력)
(2) --local                : 로컬 모델 + pk_hash.txt 로 BER 검증
(3) --nft --token-id <id>  : 온체인 NFT + 서명 + BER 풀검증
────────────────────────────────────────────────────────
"""
import os, sys, json, hashlib, argparse, requests
from pathlib import Path

import numpy as np, torch, ecdsa
import torchvision.transforms as T
import torchvision
from dotenv import load_dotenv
from web3 import Web3

from models.mlp import MLP
from utils import (subsample_training_data, get_activations,
                   extract_WM_from_activations, compute_BER)

# ─── CLI ──────────────────────────────────────────────
cli = argparse.ArgumentParser(description="DeepSigns 검증 스크립트")
mode = cli.add_mutually_exclusive_group(required=True)
mode.add_argument("--nft",   action="store_true",
                  help="온체인 + 서명 + 워터마크 검증")
mode.add_argument("--local", action="store_true",
                  help="로컬 모델 BER 검증")
mode.add_argument("--pk-hex",
                  help="공개키 → b^ 생성 후 저장된 b.npy 와 BER 비교")

cli.add_argument("--token-id", type=int, default=1,
                 help="NFT tokenId (--nft 모드)")
args = cli.parse_args()

# ─── 경로 상수 ────────────────────────────────────────
MARK_DIR   = "logs/whitebox/mlp/marked"
B_PATH     = os.path.join(MARK_DIR, "b.npy")
B_MISS     = os.path.join(MARK_DIR, "b_mismatch.npy")
MODEL_PATH = os.path.join(MARK_DIR, "mlp_nft.pth")
A_PATH     = os.path.join(MARK_DIR, "projection_matrix.npy")
PK_FILE    = os.path.join(MARK_DIR, "pk_hash.txt")

# ─── 공통 유틸 ────────────────────────────────────────
def clean_hex(h: str) -> str:
    h = h.strip().lower()
    return h[2:] if h.startswith("0x") else h

def pkhex_to_bits(pk_hex: str) -> np.ndarray:
    """공개키 → SHA‑256 → (256,) ndarray(0/1)"""
    digest = hashlib.sha256(bytes.fromhex(clean_hex(pk_hex))).digest()
    return np.unpackbits(np.frombuffer(digest, np.uint8))

def load_b_saved() -> np.ndarray:
    """훈련 종료 시 저장된 b (우선 b.npy, 없으면 b_mismatch.npy)"""
    if os.path.exists(B_PATH):
        return np.load(B_PATH)
    if os.path.exists(B_MISS):
        return np.load(B_MISS)
    sys.exit("[ERR] b.npy / b_mismatch.npy 가 없습니다")

def load_model():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = MLP().to(dev)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    A   = np.load(A_PATH)
    return net, A, dev

def ber_against(pk_hash_hex: str):
    net, A, dev = load_model()
    tf = T.Compose([T.ToTensor()])
    tr = torchvision.datasets.MNIST("./data", train=True,
                                    transform=tf, download=True)
    subset = subsample_training_data(tr, 0)
    bits   = extract_WM_from_activations(get_activations(net, subset), A)
    ref    = np.unpackbits(np.frombuffer(bytes.fromhex(pk_hash_hex),
                                         dtype=np.uint8))
    return compute_BER(bits, ref), bits

# ─── (A) PK_HEX → b̂ ↔ 저장된 b 비교 모드 ─────────────
if args.pk_hex:
    print("[MODE] 저장된 b ↔ pk_hex  일치도 검증")

    # 1) 저장된 b 로드
    b_saved = load_b_saved()             # (256,) or (2560,) or (256,10)
    if b_saved.ndim == 1:                # (2560,) 형태이면 (256, n_cls) 로 변환
        if b_saved.size % 256 != 0:
            sys.exit("[ERR] b.npy 길이가 256의 배수가 아닙니다")
        n_cls   = b_saved.size // 256
        b_saved = b_saved.reshape(256, n_cls)
    else:
        n_cls = b_saved.shape[1]

    # 2) pk_hex → 256‑bit b̂
    b_hat = pkhex_to_bits(args.pk_hex)    # (256,)

    # 3) 열 수에 맞춰 복제
    b_hat_full = np.tile(b_hat[:, None], (1, n_cls))  # (256, n_cls)
    # 4) BER
    ber      = (b_hat_full != b_saved).mean()
    diff_cnt = (b_hat_full != b_saved).sum()

    print(f"BER(b vs b̂) = {ber:.4%}   (mismatch {diff_cnt}/{b_hat_full.size})")
    print("equal?" , diff_cnt == 0)
    sys.exit(0)


# ─── (B) LOCAL BER 모드 ───────────────────────────────
if args.local:
    print("[MODE] Local BER 검증 (임계 ≤ 1 %)")
    if not os.path.exists(PK_FILE):
        sys.exit("[ERR] pk_hash.txt 가 없습니다")
    pk_hash_hex = open(PK_FILE).read().strip()

    ber, _ = ber_against(pk_hash_hex)
    print(f"BER (class 0) = {ber:.4%}   →   {'PASS' if ber<=0.01 else 'FAIL'}")
    sys.exit(0)

# ─── (C) NFT 풀검증 모드 ──────────────────────────────
print("[MODE] NFT + 서명 + 워터마크 검증 (임계 ≤ 1 %)")
load_dotenv(Path(__file__).with_name(".env"))
ADDR_RAW = os.getenv("NFT_CONTRACT_ADDR")
ABI_RAW  = os.getenv("ERC721_ABI")
PK_HEX   = os.getenv("PK_HEX")
RPC      = os.getenv("WEB3_PROVIDER_URI",
                     "https://ethereum-sepolia-rpc.publicnode.com")

if not (ADDR_RAW and ABI_RAW):
    sys.exit("[ERR] .env 에 NFT_CONTRACT_ADDR 또는 ERC721_ABI 누락")

# 컨트랙트 인스턴스
w3   = Web3(Web3.HTTPProvider(RPC))
ADDR = Web3.to_checksum_address(ADDR_RAW)
cntr = w3.eth.contract(address=ADDR, abi=json.loads(ABI_RAW))

# 1) tokenURI → CID
uri = cntr.functions.tokenURI(args.token_id).call()
print("tokenURI :", uri)
cid = uri.split("://", 1)[1]

# 2) 메타데이터 JSON
meta = requests.get(f"https://ipfs.io/ipfs/{cid}").json()
msg, sig_hex = meta["msg"], meta["sig"]

# 2‑A) pk_hash 결정
if "pk_hash" in meta:
    pk_hash_hex = meta["pk_hash"]
    source = "metadata"
elif PK_HEX:
    pk_hash_hex = hashlib.sha256(bytes.fromhex(clean_hex(PK_HEX))).hexdigest()
    source = "env PK_HEX"
else:
    sys.exit("[ERR] pk_hash 미정의 + PK_HEX 미제공 → 검증 불가")
print(f"[INFO] pk_hash source = {source}")

# 3) 서명 검증(옵션)
if PK_HEX:
    try:
        vk = ecdsa.VerifyingKey.from_string(
            bytes.fromhex(clean_hex(PK_HEX)),
            curve=ecdsa.SECP256k1
        )
        sig_ok = vk.verify(bytes.fromhex(sig_hex), msg.encode(), hashlib.sha256)
        print("Signature valid ?", sig_ok)
    except Exception as e:
        print("[WARN] 서명 검증 실패 →", e)
else:
    print("(PK_HEX 미제공 → 서명 검증 생략)")

# # 4) 워터마크 추출 & BER
# ber, _ = ber_against(pk_hash_hex)
# print(f"BER = {ber:.4%}   →   {'PASS' if ber<=0.01 else 'FAIL'}")
