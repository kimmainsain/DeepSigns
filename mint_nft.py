# -*- coding: utf-8 -*-
"""
mint_nft.py
────────────────────────────────────────────────────────
(1) logs/whitebox/mlp/marked/metadata.json 로드
(2) Pinata ‑ pinJSONToIPFS → CID → tokenURI
(3) 이미 배포된 ERC‑721 컨트랙트 mint() 호출
────────────────────────────────────────────────────────
"""
import os, json, requests, sys
from dotenv import load_dotenv
from web3 import Web3
from eth_account import Account

# ── env ────────────────────────────────────────────── #
load_dotenv()

PINATA_JWT        = os.getenv("PINATA_JWT")          # 'Bearer ' 필요 없음
RPC_URL           = os.getenv("WEB3_PROVIDER_URI")   # Sepolia RPC
PRIVATE_KEY_HEX   = os.getenv("PRIVATE_KEY_HEX")     # 64 hex
NFT_CONTRACT_ADDR = os.getenv("NFT_CONTRACT_ADDR")   # 0x…
CHAIN_ID          = int(os.getenv("CHAIN_ID", "11155111"))
META_PATH         = "logs/whitebox/mlp/marked/metadata.json"

if not all([PINATA_JWT, RPC_URL, PRIVATE_KEY_HEX, NFT_CONTRACT_ADDR]):
    sys.exit("[ERR] .env에 PINATA_JWT / WEB3_PROVIDER_URI / "
             "PRIVATE_KEY_HEX / NFT_CONTRACT_ADDR 모두 필요")

# ── 1) metadata.json ───────────────────────────────── #
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
print("[OK] metadata.json loaded")

# ── 2) Pinata → CID ────────────────────────────────── #
print("[INFO] Pinning to IPFS …")
res = requests.post(
    "https://api.pinata.cloud/pinning/pinJSONToIPFS",
    headers={"Authorization": f"Bearer {PINATA_JWT}"},
    json={"pinataContent": meta, "pinataOptions": {"cidVersion": 1}}
)
res.raise_for_status()
cid = res.json()["IpfsHash"]
token_uri = f"ipfs://{cid}"
print("    CID :", cid)

# ── 3) mint() 트랜잭션 ─────────────────────────────── #
w3   = Web3(Web3.HTTPProvider(RPC_URL))
acct = Account.from_key(bytes.fromhex(PRIVATE_KEY_HEX))
contract_addr = Web3.to_checksum_address(NFT_CONTRACT_ADDR)

# mint(to, tokenURI) 만 포함한 최소 ABI
MIN_ABI = '[{"inputs":[{"internalType":"address","name":"to","type":"address"},' \
          '{"internalType":"string","name":"tokenURI","type":"string"}],' \
          '"name":"mint","outputs":[],"stateMutability":"nonpayable","type":"function"}]'

c = w3.eth.contract(address=contract_addr, abi=MIN_ABI)

tx = c.functions.mint(acct.address, token_uri).build_transaction({
    "from":     acct.address,
    "nonce":    w3.eth.get_transaction_count(acct.address),
    "gas":      300_000,
    "gasPrice": w3.to_wei("15", "gwei"),
    "chainId":  CHAIN_ID
})

signed = acct.sign_transaction(tx)
tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
print("Using account:", acct.address, "balance:",
      w3.from_wei(w3.eth.get_balance(acct.address), "ether"), "ETH")
print("    Tx sent :", tx_hash.hex())
print("    Waiting for confirmation …")
rcpt = w3.eth.wait_for_transaction_receipt(tx_hash)
print(f"✓ confirmed  block {rcpt.blockNumber}")
print("TokenURI :", token_uri)
