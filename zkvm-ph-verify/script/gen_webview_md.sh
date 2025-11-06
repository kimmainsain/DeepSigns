#!/usr/bin/env bash
set -euo pipefail

need(){ command -v "$1" >/dev/null 2>&1 || { echo "[X] '$1' not found"; exit 1; }; }
need jq; need date

EXPLORER="${EXPLORER:-https://sepolia.etherscan.io}"

# 가장 최근 실행 폴더
RUN="${RUN:-$(ls -td runs/sepolia-* | head -n1)}"
[ -d "$RUN" ] || { echo "[X] RUN folder not found (runs/sepolia-*)"; exit 1; }

# 필수 파일들
ADDR_JSON="$RUN/addresses.json"
STATE_JSON="$RUN/state.inputs.json"
TX_REG_JSON="$RUN/reg_register.tx.json"
TX_VER_JSON="$RUN/reg_verify.tx.json"
RCP_VER_JSON="$RUN/reg_verify.rcp.json"

for f in "$ADDR_JSON" "$STATE_JSON" "$TX_REG_JSON" "$TX_VER_JSON" "$RCP_VER_JSON"; do
  [ -s "$f" ] || { echo "[X] missing: $f"; exit 1; }
done

ROUTER=$(jq -r '.router' "$ADDR_JSON")
REG=$(jq -r '.registry' "$ADDR_JSON")
PROGRAM_ID=$(jq -r '.programId' "$STATE_JSON")
PH=$(jq -r '.publicDataHash' "$STATE_JSON")
JOURNAL_DIGEST=$(jq -r '.journalDigest' "$STATE_JSON")

TX_REG=$(jq -r '.transactionHash' "$TX_REG_JSON")
TX_VER=$(jq -r '.transactionHash' "$TX_VER_JSON")

# verify 이벤트(Registry.Verified) 파싱 (없으면 빈값)
SIG_VER=$(cast keccak 'Verified(bytes32,bytes32,address)' 2>/dev/null || echo "0x")
EV_LINE=$(jq -r --arg s "$SIG_VER" 'try (.logs[] | select(.topics[0]==$s)) catch empty' "$RCP_VER_JSON")
VERIFIED_PID=$( [ -n "$EV_LINE" ] && echo "$EV_LINE" | jq -r '.data' | sed 's/^0x//' | cut -c1-64  | xargs -I{} echo 0x{} || echo "" )
VERIFIED_PDH=$( [ -n "$EV_LINE" ] && echo "$EV_LINE" | jq -r '.data' | sed 's/^0x//' | cut -c65-128 | xargs -I{} echo 0x{} || echo "" )
SUBMITTER=$( [ -n "$EV_LINE" ] && echo "$EV_LINE" | jq -r '.topics[1]' | sed 's/^0x000000000000000000000000/0x/' || echo "" )

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
OUT_MD="WEBVIEW.md"

cat > "$OUT_MD" <<EOF
# Sepolia 검증 결과(웹 뷰)

*Updated: ${TS}*

## 📌 컨트랙트 주소
- **Router (MockVerifierRouter)**: [$ROUTER]($EXPLORER/address/$ROUTER)
- **Registry**: [$REG]($EXPLORER/address/$REG)

## 🔑 입력(로컬 → 온체인)
- **programId (imageId)**: \`$PROGRAM_ID\`
- **PH (PublicDataHash)**: \`$PH\`
- **journalDigest = SHA256(PH)**: \`$JOURNAL_DIGEST\`
- **seal (EVM bytes)**: 260 bytes *(파일: \`$RUN/seal.bin\`)*

## 🧾 트랜잭션
- **register**: [$TX_REG]($EXPLORER/tx/$TX_REG)
- **verify**: [$TX_VER]($EXPLORER/tx/$TX_VER)

### Logs에서 확인할 것
**Registry.Verified(bytes32 programId, bytes32 publicDataHash, address submitter)**
- programId: \`$VERIFIED_PID\`
- publicDataHash: \`$VERIFIED_PDH\`
- submitter: \`$SUBMITTER\`

**Router.Called(bytes seal, bytes32 imageId, bytes32 journalDigest)**
- imageId = \`$PROGRAM_ID\` 와 일치
- journalDigest = \`$JOURNAL_DIGEST\` 와 일치
- seal 길이 = 260 bytes (입력과 동일)

## 🗂️ 산출물 → 결과물 맵
| 파일 | 생성 데이터 | 온체인/웹에서 보이는 결과 |
|---|---|---|
| \`$STATE_JSON\` | programId, PH, SHA256(PH), seal(hex) | Tx Input / Logs 값과 일치 |
| \`$RUN/seal.bin\` | 260B EVM seal | Router.Called의 \`seal\`(bytes) |
| \`$RUN/router.out.json\` | Router 주소 | [$ROUTER]($EXPLORER/address/$ROUTER) |
| \`$RUN/reg.out.json\` | Registry 주소 | [$REG]($EXPLORER/address/$REG) |
| \`$TX_REG_JSON\` | register Tx hash | [$TX_REG]($EXPLORER/tx/$TX_REG) |
| \`$TX_VER_JSON\` | verify Tx hash | [$TX_VER]($EXPLORER/tx/$TX_VER) |
| \`$RUN/reg_verify.rcp.json\` | Verified / Called 이벤트 원본 | 각 Tx의 **Logs** 탭 |

## 🔍 검증 포인트(사람 눈으로)
1. **register Tx**의 Input 디코드에서 \`programId\`, \`publicDataHash\` 확인  
2. **verify Tx**의 Input 디코드에서 \`programId\`, \`publicDataHash\`, \`seal\` 확인  
3. **Registry Logs** → **Verified** 이벤트의 두 값이 위 입력과 **완전히 동일**한지 확인  
4. **Router Logs** → **Called** 이벤트의 \`imageId\`=\`programId\`, \`journalDigest\`=\`SHA256(PH)\` 확인

> 💡 지금은 **MockRouter**라 암호학적 검증은 생략(이벤트만).  
> 실제 검증 라우터로 바뀌면 동일한 흐름으로 **진짜 증명 검증**이 동작.

---
EOF

echo "[✓] Wrote $OUT_MD (links point to $EXPLORER)"

