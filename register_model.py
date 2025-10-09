# register_sig_only_digest.py
import os, sys, json, hashlib, ecdsa
from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv

load_dotenv()
RPC_URL       = os.getenv("RPC_URL") or os.getenv("WEB3_PROVIDER_URI")
REGISTRY_ADDR = os.getenv("REGISTRY_ADDR")   # WMAnchorSigOnly 주소
SK_HEX        = os.getenv("PRIVATE_KEY_HEX") # 64-hex (no 0x)
if not (RPC_URL and REGISTRY_ADDR and SK_HEX):
    sys.exit("[ERR] RPC_URL/REGISTRY_ADDR/PRIVATE_KEY_HEX 필요")

w3   = Web3(Web3.HTTPProvider(RPC_URL))
acct = Account.from_key(bytes.fromhex(SK_HEX))
print("Connected:", w3.is_connected(), "chainId:", w3.eth.chain_id)
print("Using address:", acct.address)

# 1) 모델 해시(digest, 32B) 계산
MODEL_PATH = "logs/whitebox/mlp/marked/mlp_nft.pth"
with open(MODEL_PATH, "rb") as f:
    model_hash_digest = hashlib.sha256(f.read()).digest()  # bytes(32)

# 2) ECDSA 서명: digest 전용
sk  = ecdsa.SigningKey.from_string(bytes.fromhex(SK_HEX), curve=ecdsa.SECP256k1)
# DER 서명으로 고정(가변 길이  ~70B), digest에 대해 결정적 서명
sig = sk.sign_digest_deterministic(
    model_hash_digest,
    hashfunc=hashlib.sha256,
    sigencode=ecdsa.util.sigencode_der
)

# 3) 컨트랙트 호출
ABI = json.loads(r'''
[
  {"inputs":[{"internalType":"bytes","name":"sig","type":"bytes"}],
   "name":"upsertSig","outputs":[],"stateMutability":"nonpayable","type":"function"},
  {"inputs":[{"internalType":"address","name":"owner","type":"address"},
             {"internalType":"bytes32","name":"sigCommit","type":"bytes32"}],
   "name":"getEntryBySigCommit","outputs":[{"components":[
     {"internalType":"bytes32","name":"sigCommit","type":"bytes32"},
     {"internalType":"uint64","name":"version","type":"uint64"},
     {"internalType":"uint64","name":"blockNumber","type":"uint64"},
     {"internalType":"bool","name":"revoked","type":"bool"}],
     "internalType":"struct WMAnchorSigOnly.Entry","name":"","type":"tuple"}],
   "stateMutability":"view","type":"function"}
]
''')
c = w3.eth.contract(address=w3.to_checksum_address(REGISTRY_ADDR), abi=ABI)

latest   = w3.eth.get_block('latest')
base_fee = latest.get('baseFeePerGas', 0)
try:
    priority = int(w3.eth.max_priority_fee)
except Exception:
    priority = int(2e9)
max_fee  = int(base_fee * 2 + priority)

tx = c.functions.upsertSig(sig).build_transaction({
    "from": acct.address,
    "nonce": w3.eth.get_transaction_count(acct.address, 'pending'),
    "chainId": w3.eth.chain_id,
    "maxFeePerGas": max_fee,
    "maxPriorityFeePerGas": priority,
})
tx["gas"] = w3.eth.estimate_gas(tx)

signed = acct.sign_transaction(tx)
raw_tx = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None)
txh = w3.eth.send_raw_transaction(raw_tx)
rcp = w3.eth.wait_for_transaction_receipt(txh, timeout=300, poll_latency=3)

print("tx:", txh.hex(), "status:", rcp.status)
print("sig(hex):", sig.hex())
print("sigCommit:", Web3.keccak(sig).hex())
print("model_hash_hex:", model_hash_digest.hex())  # 오프체인 보관용
