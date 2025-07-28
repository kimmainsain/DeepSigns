# -*- coding: utf-8 -*-
"""
mint_nft.py
────────────────────────────────────────────────────────
(1) pk_hash.txt + **mlp_nft.pth 해시**로 메타데이터 생성
(2) Pinata pinJSONToIPFS → CID → tokenURI
(3) 컨트랙트 mint() 호출
────────────────────────────────────────────────────────
"""
import os, json, sys, hashlib, datetime, requests
from dotenv import load_dotenv
from web3 import Web3
from eth_account import Account
import ecdsa

load_dotenv()

PINATA_JWT        = os.getenv("PINATA_JWT")
RPC_URL           = os.getenv("WEB3_PROVIDER_URI")
PRIVATE_KEY_HEX   = os.getenv("PRIVATE_KEY_HEX")
NFT_CONTRACT_ADDR = os.getenv("NFT_CONTRACT_ADDR")
ERC721_ABI_RAW    = os.getenv("ERC721_ABI")
CHAIN_ID          = int(os.getenv("CHAIN_ID", "11155111"))

if not all([PINATA_JWT, RPC_URL, PRIVATE_KEY_HEX, NFT_CONTRACT_ADDR, ERC721_ABI_RAW]):
    sys.exit("[ERR] .env 항목 누락")

# ── 1) 산출물 로드 ───────────────────────────────────── #
marked_dir = "logs/whitebox/mlp/marked"

with open(f"{marked_dir}/pk_hash.txt") as f:
    pk_hash = f.read().strip()

### BEGIN PATCH – 모델 SHA‑256 해시 계산 ##################
model_path = f"{marked_dir}/mlp_nft.pth"
if not os.path.exists(model_path):
    sys.exit("[ERR] mlp_nft.pth 가 없습니다")

with open(model_path, "rb") as f:
    model_hash = hashlib.sha256(f.read()).hexdigest()
### END PATCH ############################################

# 1‑1) 메시지 생성
ts_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
#  ↓ public‑key 대신 model_hash 포함
msg = f"ModelHash={model_hash};Timestamp={ts_iso}"

# 1‑2) ECDSA 개인키 로드 & 서명 (deterministic)
sk = ecdsa.SigningKey.from_string(
    bytes.fromhex(PRIVATE_KEY_HEX),
    curve=ecdsa.SECP256k1
)
sig = sk.sign_deterministic(msg.encode(), hashfunc=hashlib.sha256)

### BEGIN PATCH – 메타데이터 구조 수정 ####################
metadata = {
    "msg":        msg,
    "sig":        sig.hex(),
    "model_hash": model_hash
}
### END PATCH ############################################

# ── 2) Pinata → CID ────────────────────────────────── #
print("[INFO] Pinning metadata to IPFS …")
res = requests.post(
    "https://api.pinata.cloud/pinning/pinJSONToIPFS",
    headers={"Authorization": f"Bearer {PINATA_JWT}"},
    json={"pinataContent": metadata, "pinataOptions": {"cidVersion": 1}}
)
res.raise_for_status()
cid = res.json()["IpfsHash"]
token_uri = f"ipfs://{cid}"
print("CID :", cid)

# ── 3) Web3 초기화 & 컨트랙트 인스턴스 ─────────────── #
w3   = Web3(Web3.HTTPProvider(RPC_URL))
acct = Account.from_key(bytes.fromhex(PRIVATE_KEY_HEX))
cntr = w3.eth.contract(
    address=Web3.to_checksum_address(NFT_CONTRACT_ADDR),
    abi=json.loads(ERC721_ABI_RAW)
)

# ── 4) mint() 트랜잭션 생성 ────────────────────────── #
tx = cntr.functions.mint(acct.address, token_uri).build_transaction({
    "from":     acct.address,
    "nonce":    w3.eth.get_transaction_count(acct.address),
    "gas":      300_000,
    "gasPrice": w3.to_wei("15", "gwei"),
    "chainId":  CHAIN_ID
})

# ── 5) 서명 → 전송 → 컨펌 ───────────────────────────── #
signed_tx = acct.sign_transaction(tx)
tx_hash   = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
print("Tx sent : ", tx_hash.hex(), "… waiting …")
rcpt = w3.eth.wait_for_transaction_receipt(tx_hash)
print(f"✓ confirmed in block {rcpt.blockNumber}")
print("TokenURI :", token_uri)
