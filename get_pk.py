from dotenv import load_dotenv
import os
from eth_account import Account

load_dotenv()
PRIVATE_KEY_HEX = os.getenv("PRIVATE_KEY_HEX")
if not PRIVATE_KEY_HEX:
    raise Exception(".env에 PRIVATE_KEY_HEX가 없습니다.")
sk = bytes.fromhex(PRIVATE_KEY_HEX)
pk_hex = "04" + Account.from_key(sk)._key_obj.public_key.to_bytes().hex()
print("PK =", pk_hex)
