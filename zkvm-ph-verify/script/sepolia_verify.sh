#!/usr/bin/env bash
# Registry(2-arg register) + Real Verifier Router 경유
# Flow: (1) register(programId, PH) -> (2) verify(programId, PH, seal, journal=PH)
# Needs: forge, cast, jq, xxd, openssl
set -euo pipefail

need(){ command -v "$1" >/dev/null 2>&1 || { echo "[X] '$1' not found"; exit 1; }; }
need forge; need cast; need jq; need xxd; need openssl

# ===== 필요한 환경변수 =====
: "${RPC_URL:?set RPC_URL (Sepolia HTTP endpoint)}"
: "${PRIVATE_KEY_HEX:?set PRIVATE_KEY_HEX (64 hex)}"
: "${PROGRAM_ID:?set PROGRAM_ID (0x + 32 bytes hex)}"     # imageId/programId (bytes32)
: "${PH:?set PH (0x + 32 bytes hex)}"                     # Public Data Hash(= journal 원문 32바이트)
: "${REG:?set REG (deployed Registry addr)}"
# 최소 하나는 있어야 함: SEAL_HEX 또는 SEAL_BIN
if [ -z "${SEAL_HEX:-}" ] && [ -z "${SEAL_BIN:-}" ]; then
  echo "[X] set SEAL_HEX or SEAL_BIN"; exit 1
fi

PRIVATE_KEY="0x${PRIVATE_KEY_HEX#0x}"
FROM="$(cast wallet address --private-key "$PRIVATE_KEY")"

RUN="runs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN"

# ===== 1) 입력 정리 =====
# JOURNAL_DIGEST: env 우선, 없으면 PH -> SHA-256
if [ -n "${JOURNAL_DIGEST:-}" ]; then
  echo "[*] JOURNAL_DIGEST (env) = $JOURNAL_DIGEST"
else
  PH_CLEAN="$(echo -n "$PH" | tr -d ' \t\r\n')"
  PH_BIN="$RUN/ph.bin"
  echo -n "${PH_CLEAN#0x}" | xxd -r -p > "$PH_BIN"
  JOURNAL_DIGEST="0x$(openssl dgst -sha256 -binary "$PH_BIN" | xxd -p -c 256)"
fi

# SEAL_HEX: env 우선, 없으면 파일에서 읽기
if [ -n "${SEAL_HEX:-}" ]; then
  case "$SEAL_HEX" in 0x*) ;; *) SEAL_HEX="0x$SEAL_HEX" ;; esac
elif [ -n "${SEAL_BIN:-}" ] && [ -f "$SEAL_BIN" ]; then
  SEAL_HEX="0x$(xxd -p -c 0 "$SEAL_BIN")"
else
  echo "[X] SEAL_BIN not found: ${SEAL_BIN:-<unset>}"; exit 1
fi

# 길이 표기
if [ -n "${SEAL_BIN:-}" ] && [ -f "$SEAL_BIN" ]; then
  SEAL_LEN=$(stat -c%s "$SEAL_BIN" 2>/dev/null || wc -c < "$SEAL_BIN")
else
  SEAL_LEN=$(( ( ${#SEAL_HEX} - 2 ) / 2 ))
fi

echo "[*] FROM=$FROM"
echo "[*] PROGRAM_ID=$PROGRAM_ID"
echo "[*] PH(publicDataHash RAW)= $PH"
echo "[*] JOURNAL_DIGEST(SHA256(PH))= $JOURNAL_DIGEST"
echo "[*] SEAL_LEN=${SEAL_LEN} bytes"

# ===== helper: 안전하게 트랜잭션 전송해서 해시만 뽑기 =====
send_tx() {
  # 사용법: send_tx <to> <sig> [args...]
  local to="$1"; shift
  local out
  out="$(cast send "$to" "$@" --rpc-url "$RPC_URL" --private-key "$PRIVATE_KEY" --json)"
  echo "$out" > "$RUN/last.send.json"
  jq -r '.transactionHash' <<<"$out"
}

# ===== 2) 앵커 등록 (Registry.register(bytes32,bytes32)) =====
REG_HASH="$(send_tx "$REG" "register(bytes32,bytes32)" "$PROGRAM_ID" "$PH")"
echo "[*] register txHash=$REG_HASH"
cast receipt "$REG_HASH" --rpc-url "$RPC_URL" --json > "$RUN/reg.rcp.json"
REG_STATUS="$(jq -r '.status' "$RUN/reg.rcp.json")"
REG_GAS="$(jq -r '.gasUsed' "$RUN/reg.rcp.json")"
REG_PRICE="$(jq -r '.effectiveGasPrice' "$RUN/reg.rcp.json")"
echo "[*] register status=$REG_STATUS gasUsed=$REG_GAS"
if [ "$REG_STATUS" != "0x1" ]; then
  echo "[X] register reverted"; exit 1
fi

# ===== 3) 실제 검증 (Registry.verify) =====
# verify(bytes32 programId, bytes32 publicDataHash, bytes proofSeal, bytes journal=PH)
VER_HASH="$(send_tx "$REG" "verify(bytes32,bytes32,bytes,bytes)" "$PROGRAM_ID" "$PH" "$SEAL_HEX" "$PH")"
echo "[*] verify txHash=$VER_HASH"
cast receipt "$VER_HASH" --rpc-url "$RPC_URL" --json > "$RUN/ver.rcp.json"
VER_STATUS="$(jq -r '.status' "$RUN/ver.rcp.json")"
VER_GAS="$(jq -r '.gasUsed' "$RUN/ver.rcp.json")"
VER_PRICE="$(jq -r '.effectiveGasPrice' "$RUN/ver.rcp.json")"
echo "[*] verify status=$VER_STATUS gasUsed=$VER_GAS"
if [ "$VER_STATUS" != "0x1" ]; then
  echo "[X] verify reverted (check programId / PH / seal set)"; exit 1
fi

# ===== 4) 가스/비용 합계 출력 =====
VGAS_DEC="$(cast --to-dec "$VER_GAS")"
VPRX_DEC="$(cast --to-dec "$VER_PRICE")"
RGAS_DEC="$(cast --to-dec "$REG_GAS")"
RPRX_DEC="$(cast --to-dec "$REG_PRICE")"
VWEI=$((VGAS_DEC*VPRX_DEC))
RWEI=$((RGAS_DEC*RPRX_DEC))
TWEI=$((VWEI+RWEI))

echo "== Gas Summary =="
echo "register gasUsed: $RGAS_DEC, price: $(cast --from-wei $RPRX_DEC gwei) gwei, cost: $(cast --from-wei $RWEI ether) ETH"
echo "verify   gasUsed: $VGAS_DEC, price: $(cast --from-wei $VPRX_DEC gwei) gwei, cost: $(cast --from-wei $VWEI ether) ETH"
echo "total    gasUsed: $((VGAS_DEC+RGAS_DEC))"
echo "total    cost:    $(cast --from-wei $TWEI ether) ETH"

echo "[✓] Done. artifacts → $RUN"

