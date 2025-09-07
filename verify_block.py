# verify_sig_only_digest.py
import os, sys, hashlib, json
from web3 import Web3
from hexbytes import HexBytes
import ecdsa
from dotenv import load_dotenv

load_dotenv()
RPC_URL       = os.getenv("RPC_URL") or os.getenv("WEB3_PROVIDER_URI") or "https://ethereum-sepolia-rpc.publicnode.com"
REGISTRY_ADDR = os.getenv("REGISTRY_ADDR")
TX_HASH       = os.getenv("TX_HASH") or input("tx hash : ").strip()
PK_RAW64_HEX  = os.getenv("PK_RAW64_HEX") or input("pk_raw64 hex : ").strip()
MODEL_HASH_HEX= os.getenv("MODEL_HASH_HEX") or input("model hash hex : ").strip()

# --- 입력 정규화 ---
if not TX_HASH.startswith("0x"):
    TX_HASH = "0x" + TX_HASH

# 공개키: raw64(64B)로 정규화 (언컴프레스트 65B면 0x04 제거)
pkh = PK_RAW64_HEX.lower().replace("0x", "")
if len(pkh) == 130 and pkh.startswith("04"):
    pkh = pkh[2:]
if len(pkh) != 128:
    sys.exit(f"[ERR] pk_raw64 must be 64 bytes (128 hex). got {len(pkh)//2} bytes")
pk_raw64 = bytes.fromhex(pkh)

# 모델 해시: 32바이트 hex
mh = MODEL_HASH_HEX.lower().replace("0x","")
if len(mh) != 64:
    sys.exit(f"[ERR] model hash must be 32 bytes (64 hex). got {len(mh)//2} bytes")
try:
    model_hash_digest = bytes.fromhex(mh)
except ValueError:
    sys.exit("[ERR] model hash is not valid hex")

# --- RPC 연결 ---
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    sys.exit(f"[ERR] cannot connect RPC: {RPC_URL}")

# 1) tx receipt에서 sig만 추출
topic0 = Web3.keccak(text="UpsertSig(address,bytes,uint64)")
rcp = w3.eth.get_transaction_receipt(TX_HASH)
logs = [lg for lg in rcp.logs if lg.topics and lg.topics[0] == topic0]
if not logs:
    sys.exit("[ERR] UpsertSig event not found in tx")
log = logs[0]
owner = Web3.to_checksum_address("0x" + log.topics[1].hex()[-40:])

data = HexBytes(log.data)
off_sig = int.from_bytes(data[0:32], "big")
version = int.from_bytes(data[32:64], "big")
sig_len = int.from_bytes(data[off_sig:off_sig+32], "big")
sig     = bytes(data[off_sig+32:off_sig+32+sig_len])  # DER

# (선택) 온체인 커밋 대조
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
sig_commit_local = Web3.keccak(sig)
entry = c.functions.getEntryBySigCommit(owner, sig_commit_local).call()
assert entry[0] == sig_commit_local and not entry[3], "onchain commit mismatch or revoked"

# 3) 공개키로 digest 검증 (DER + verify_digest)
vk = ecdsa.VerifyingKey.from_string(pk_raw64, curve=ecdsa.SECP256k1)
ok = vk.verify_digest(sig, model_hash_digest, sigdecode=ecdsa.util.sigdecode_der)

print("owner:", owner)
print("signature valid?:", ok)
print("version:", version)
