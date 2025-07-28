# get_pk.py  ──────────────────────────────────────────
from dotenv import load_dotenv
import os, ecdsa

load_dotenv()
sk_hex = os.getenv("PRIVATE_KEY_HEX")
if not sk_hex:
    raise SystemExit(".env에 PRIVATE_KEY_HEX가 없습니다.")

# 1) 개인키 → ecdsa.SigningKey 객체
sk = ecdsa.SigningKey.from_string(bytes.fromhex(sk_hex),
                                  curve=ecdsa.SECP256k1)

# 2) 공개키 (언컴프레스 65 바이트)  →  130‑hex
pk_hex = sk.get_verifying_key()\
           .to_string("uncompressed")\
           .hex()

print("PK =", pk_hex)
