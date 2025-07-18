# verify.py  ───────────────────────────────────────────
import os, json, hashlib, requests, numpy as np, torch, ecdsa
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv

# DeepSigns util & model
from models.mlp import MLP
import torchvision, torchvision.transforms as T
from utils import (subsample_training_data, get_activations,
                   extract_WM_from_activations, compute_BER)

# ── .env 로드 ─────────────────────────────────────────
load_dotenv(Path(__file__).with_name(".env"))

ADDR_RAW  = os.getenv("NFT_CONTRACT_ADDR")
ABI_RAW   = os.getenv("ERC721_ABI")
PK_HEX    = os.getenv("PK_HEX")           # 없으면 None
RPC       = os.getenv("WEB3_PROVIDER_URI",
                      "https://ethereum-sepolia-rpc.publicnode.com")
if not ADDR_RAW:
    raise SystemExit("[ERR] .env 에 NFT_CONTRACT_ADDR 가 없습니다!")
if not ABI_RAW:
    raise SystemExit("[ERR] .env 에 ERC721_ABI 가 없습니다!")

try:
    ABI = json.loads(ABI_RAW)
    if not isinstance(ABI, list):
        raise ValueError
except Exception:
    raise SystemExit("[ERR] ERC721_ABI 는 JSON 배열 한 줄이어야 합니다.")

ADDR = Web3.to_checksum_address(ADDR_RAW)
w3   = Web3(Web3.HTTPProvider(RPC))
c    = w3.eth.contract(address=ADDR, abi=ABI)

# ── 1) tokenURI ──────────────────────────────────────
token_id = 1
uri = c.functions.tokenURI(token_id).call()
print("tokenURI:", uri)
cid = uri.split("://", 1)[1]

# ── 2) metadata.json 가져오기 ────────────────────────
meta = requests.get(f"https://ipfs.io/ipfs/{cid}").json()
msg, sig_hex, pk_hash_hex = meta["msg"], meta["sig"], meta["pk_hash"]

# ── 3) (옵션) 서명 검증 ──────────────────────────────
if PK_HEX:
    try:
        vk_bytes = bytes.fromhex(PK_HEX[2:] if PK_HEX.startswith("04") else PK_HEX)
        pk = ecdsa.VerifyingKey.from_string(vk_bytes, curve=ecdsa.SECP256k1)
        ok = pk.verify(bytes.fromhex(sig_hex), msg.encode(),
                       hashfunc=hashlib.sha256)
        print("Signature valid?", ok)
    except Exception as e:
        print("[WARN] PK_HEX 주어진 공개키 검증 실패 →", e)
else:
    print("(PK_HEX 없어서 서명 검증 단계 건너뜀)")

# ── 4) 모델 워터마크 추출 → pk_hash 비교 ─────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = MLP().to(device)
model.load_state_dict(torch.load('logs/whitebox/mlp/marked/mlp_nft.pth',
                                 map_location=device))
A = np.load('logs/whitebox/mlp/marked/projection_matrix.npy')

tf = T.Compose([T.ToTensor()])
trainset = torchvision.datasets.MNIST("./data", train=True,
                                      transform=tf, download=True)
subset   = subsample_training_data(trainset, 0)
activs   = get_activations(model, subset)
bits_dec = extract_WM_from_activations(activs, A)

bits_ref = np.unpackbits(np.frombuffer(bytes.fromhex(pk_hash_hex),
                                       dtype=np.uint8))
ber = compute_BER(bits_dec, bits_ref)
print("BER:", ber)
print("Hash match?", np.packbits(bits_dec).tobytes().hex() == pk_hash_hex)
