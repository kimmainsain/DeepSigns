#!/usr/bin/env bash
# One-shot: receipt.bin → onchain_prep → (Verifier) → Router → Registry → register/verify (Sepolia)
# 위치: DeepSigns/zkvm-ph-verify
set -euo pipefail

need(){ command -v "$1" >/dev/null 2>&1 || { echo "[X] '$1' not found"; exit 1; }; }
need forge; need cast; need jq; need xxd; need cargo

# ===== 0) 필수 환경 =====
: "${RPC_URL:?set RPC_URL (Sepolia HTTP endpoint)}"
: "${PRIVATE_KEY_HEX:?set PRIVATE_KEY_HEX (64 hex)}"
: "${PH:?set PH (0x + 64 hex, guest journal raw)}"

PRIVATE_KEY="0x${PRIVATE_KEY_HEX#0x}"
PRIVATE_KEY="$(echo -n "$PRIVATE_KEY" | tr -d ' \t\r\n')"
FROM="$(cast wallet address --private-key "$PRIVATE_KEY")" || { echo "[X] invalid PRIVATE_KEY"; exit 1; }
CHAIN_ID="$(cast chain-id --rpc-url "$RPC_URL" 2>/dev/null || true)"
[[ "$CHAIN_ID" == "11155111" ]] || { echo "[X] RPC_URL must be Sepolia (chain-id=11155111), got '$CHAIN_ID'"; exit 1; }

echo "[i] FROM=$FROM"
echo "[i] RPC_URL=$RPC_URL"
echo "[i] CHAIN_ID=$CHAIN_ID"

# ===== 1) 경로/입력 파일 =====
ROOT="$(pwd)"
RUST_WS="${RUST_WS:-$ROOT/../zkvm}"
RECEIPT="${RECEIPT:-$RUST_WS/data/receipt.bin}"
[[ -f "$RECEIPT" ]] || { echo "[X] receipt not found: $RECEIPT"; exit 1; }

RUN="${RUN:-runs/sepolia-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN"
echo "RUN=$RUN" | tee "$RUN/_run.info.txt"

# ===== 2) onchain_prep + receipt_info =====
echo "[*] onchain_prep..."
cargo run --manifest-path "$RUST_WS/Cargo.toml" -p host --bin onchain_prep --release -- \
  --receipt "$RECEIPT" \
  --ph "$PH" \
  --out-seal "$RUN/seal.bin" | tee "$RUN/onchain_prep.out.txt"

echo "[*] receipt_info..."
cargo run --manifest-path "$RUST_WS/Cargo.toml" -p host --bin receipt_info --release -- \
  --receipt "$RECEIPT" | tee "$RUN/receipt_info.txt" || true

# PROGRAM_ID 자동 추출
if [[ -z "${PROGRAM_ID:-}" ]]; then
  if [[ -f "$RUN/onchain_prep.out.txt" ]]; then
    PROGRAM_ID=$(
      awk '/programId \(imageId\)/ {for(i=1;i<=NF;i++) if ($i ~ /^0x[0-9a-fA-F]{64}$/) print $i}' \
      "$RUN/onchain_prep.out.txt" | head -n1
    )
  fi
fi
if [[ -z "${PROGRAM_ID:-}" && -s "$RUN/receipt_info.txt" ]]; then
  PROGRAM_ID=$(
    grep -Eoi '0x[0-9a-fA-F]{64}' "$RUN/receipt_info.txt" | head -n1
  )
fi
: "${PROGRAM_ID:?PROGRAM_ID not found. Set PROGRAM_ID=0x<64hex>}"

# JOURNAL_DIGEST = SHA-256(PH 원문)
if command -v sha256sum >/dev/null 2>&1; then
  JOURNAL_DIGEST="0x$(echo -n "${PH#0x}" | xxd -r -p | sha256sum | awk '{print $1}')"
else
  JOURNAL_DIGEST="0x$(echo -n "${PH#0x}" | xxd -r -p | shasum -a 256 | awk '{print $1}')"
fi

SEAL_HEX="0x$(xxd -p -c 999999 "$RUN/seal.bin")"
SEAL_LEN=$(( ( ${#SEAL_HEX} - 2 ) / 2 ))
echo "seal bytes = $SEAL_LEN" | tee -a "$RUN/onchain_prep.out.txt"
[[ "$SEAL_LEN" -eq 260 ]] || echo "[!] WARN: seal length != 260"

# state 파일: publicDataHash=sha256(journal), journal=PH(raw)
jq -n --arg programId "$PROGRAM_ID" \
      --arg publicDataHash "$JOURNAL_DIGEST" \
      --arg journal "$PH" \
      --arg seal "$SEAL_HEX" \
      '{programId:$programId, publicDataHash:$publicDataHash, journal:$journal, sealHex:$seal}' \
      > "$RUN/state.inputs.json"

# ===== helpers =====
extract_addr () {
  local f="$1"
  jq -r 'select(.contract_address!=null) | .contract_address' "$f" 2>/dev/null || true
  jq -r 'select(.returns!=null) | .returns[]? | .value? // empty' "$f" 2>/dev/null || true
  jq -r 'select(.logs!=null) | .logs[] | capture("(?<addr>0x[0-9a-fA-F]{40})").addr' "$f" 2>/dev/null || true
  local t s
  t=$(jq -r 'select(.transactions!=null) | .transactions' "$f" 2>/dev/null || true)
  s=$(jq -r 'select(.sensitive!=null)    | .sensitive'    "$f" 2>/dev/null || true)
  for rf in "$t" "$s"; do
    [[ -n "${rf:-}" && -f "$rf" ]] && jq -r '.. | .contractAddress? | select(.!=null)' "$rf" 2>/dev/null || true
  done
}
addr_candidates () {
  sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//' \
  | grep -Eoi '0x[0-9a-fA-F]{40}' \
  | awk '{k=tolower($0)} !seen[k]++'
}
wait_for_code(){
  local addr="$1"; local tries="${2:-60}"; local delay="${3:-1}"
  for ((i=0;i<tries;i++)); do
    local code
    code="$(cast code "$addr" --rpc-url "$RPC_URL" 2>/dev/null || true)"
    if [[ -n "$code" && "$code" != "0x" ]]; then return 0; fi
    sleep "$delay"
  done
  return 1
}

# ===== 3-0) Verifier (없으면 배포 / 있으면 재사용) =====
if [[ -n "${VERIFIER:-}" ]]; then
  VERIFIER="$(echo -n "$VERIFIER" | tr -d ' \t\r\n')"
  echo "[*] Reusing VERIFIER=$VERIFIER"
else
  echo "[*] DeployVerifier.s.sol (Sepolia)..."
  forge script script/DeployVerifier.s.sol \
    --rpc-url "$RPC_URL" --private-key "$PRIVATE_KEY" \
    --broadcast -vv --json | tee "$RUN/verifier.out.json"
  VERIFIER=""
  if [[ -f "$RUN/verifier.out.json" ]]; then
    mapfile -t VC < <(extract_addr "$RUN/verifier.out.json" | addr_candidates | tac)
    for a in "${VC[@]}"; do
      if wait_for_code "$a" 60 1; then VERIFIER="$a"; break; fi
    done
  fi
  : "${VERIFIER:?Verifier address not found}"
fi

# ===== 3-1) Router (지정 없으면 배포; VERIFIER 주입) =====
if [[ -n "${ROUTER:-}" ]]; then
  ROUTER="$(echo -n "$ROUTER" | tr -d ' \t\r\n')"
  echo "[*] Reusing ROUTER=$ROUTER"
else
  echo "[*] DeployRouter.s.sol (Sepolia)..."
  VERIFIER="$VERIFIER" forge script script/DeployRouter.s.sol \
    --rpc-url "$RPC_URL" \
    --private-key "$PRIVATE_KEY" \
    --broadcast -vv --json | tee "$RUN/router.out.json"

  ROUTER=""
  mapfile -t CANDS < <(extract_addr "$RUN/router.out.json" | addr_candidates | tac)
  for a in "${CANDS[@]}"; do
    if wait_for_code "$a" 60 1; then ROUTER="$a"; break; fi
  done
  : "${ROUTER:?Router address not found}"
fi

echo "ROUTER=$ROUTER" | tee -a "$RUN/addresses.txt"
cast code "$ROUTER" --rpc-url "$RPC_URL" | sed -n '1,3p' | tee "$RUN/router.code.head.txt"

# ===== 3-2) Registry (항상 배포; ROUTER 주입) =====
echo "[*] DeployRegistry.s.sol (Sepolia)..."
export ROUTER
forge script script/DeployRegistry.s.sol \
  --rpc-url "$RPC_URL" \
  --private-key "$PRIVATE_KEY" \
  --broadcast -vv --json | tee "$RUN/reg.out.json"

REG=""
mapfile -t CANDS2 < <(extract_addr "$RUN/reg.out.json" | addr_candidates | tac)
for a in "${CANDS2[@]}"; do
  if wait_for_code "$a" 60 1; then REG="$a"; break; fi
done
: "${REG:?Registry address not found}"

echo "REG=$REG" | tee -a "$RUN/addresses.txt"
cast code "$REG" --rpc-url "$RPC_URL" | sed -n '1,3p' | tee "$RUN/registry.code.head.txt"

jq -n --arg verifier "$VERIFIER" --arg router "$ROUTER" --arg reg "$REG" \
      '{verifier:$verifier, router:$router, registry:$reg}' > "$RUN/addresses.json"
cat > "$RUN/export.env" <<EOF
# reuse addresses from this run
export VERIFIER="$VERIFIER"
export ROUTER="$ROUTER"
export REG="$REG"
EOF

# ===== 4) register → verify =====
echo "[*] register(programId, PH)..."
cast send "$REG" "register(bytes32,bytes32)" \
  "$PROGRAM_ID" "$JOURNAL_DIGEST" \
  --rpc-url "$RPC_URL" --private-key "$PRIVATE_KEY" \
  --json | tee "$RUN/reg_register.tx.json"

REG_TX="$(jq -r '.transactionHash' "$RUN/reg_register.tx.json")"
cast receipt "$REG_TX" --rpc-url "$RPC_URL" --json | tee "$RUN/reg_register.rcp.json"

echo "[*] verify(programId, publicDataHash, seal, journalRawPH)..."
cast send "$REG" "verify(bytes32,bytes32,bytes,bytes)" \
  "$PROGRAM_ID" "$JOURNAL_DIGEST" "$SEAL_HEX" "$PH" \
  --gas-limit 1200000 \
  --rpc-url "$RPC_URL" --private-key "$PRIVATE_KEY" \
  --json | tee "$RUN/reg_verify.tx.json"

VER_TX="$(jq -r '.transactionHash' "$RUN/reg_verify.tx.json")"
cast receipt "$VER_TX" --rpc-url "$RPC_URL" --json | tee "$RUN/reg_verify.rcp.json"

# ===== 5) 이벤트/일치성/가스 CSV =====
SIG_VERIFIED="$(cast keccak 'Verified(bytes32,bytes32,address)')"
jq -r --arg s "$SIG_VERIFIED" '.logs[] | select(.topics[0]==$s)' "$RUN/reg_verify.rcp.json" > "$RUN/ev.Registry.Verified.json" || true

if [[ -s "$RUN/ev.Registry.Verified.json" ]]; then
  DATA="$(jq -r '.data' "$RUN/ev.Registry.Verified.json" | head -n1)"
  H="${DATA#0x}"; PID="0x${H:0:64}"; PDH="0x${H:64:64}"
  {
    echo "Verified.programId      = $PID"
    echo "Verified.publicDataHash = $PDH"
    echo "programId check         : $([[ "${PID,,}" == "${PROGRAM_ID,,}" ]] && echo OK || echo MISMATCH)"
    echo "PDH check               : $([[ "${PDH,,}" == "${JOURNAL_DIGEST,,}" ]] && echo OK || echo MISMATCH)"
  } | tee -a "$RUN/verify.parsed.txt"
else
  echo "[!] Registry.Verified event not found (컨트랙트 구현에 따라 정상일 수 있음)" | tee -a "$RUN/verify.parsed.txt"
fi

CSV="$RUN/gas.csv"
echo "timestamp,network,tx,method,gasUsed,effectiveGasPriceWei,costWei,programId,verifier,router,registry,sealBytes" > "$CSV"

REG_GAS="$(jq -r '.gasUsed' "$RUN/reg_register.rcp.json")"
REG_EGP="$(jq -r '.effectiveGasPrice' "$RUN/reg_register.rcp.json")"
VER_GAS="$(jq -r '.gasUsed' "$RUN/reg_verify.rcp.json")"
VER_EGP="$(jq -r '.effectiveGasPrice' "$RUN/reg_verify.rcp.json")"
REG_COST=$(( REG_GAS * REG_EGP ))
VER_COST=$(( VER_GAS * VER_EGP ))
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "$TS,sepolia,$REG_TX,register,$REG_GAS,$REG_EGP,$REG_COST,$PROGRAM_ID,$VERIFIER,$ROUTER,$REG,$SEAL_LEN" >> "$CSV"
echo "$TS,sepolia,$VER_TX,verify,$VER_GAS,$VER_EGP,$VER_COST,$PROGRAM_ID,$VERIFIER,$ROUTER,$REG,$SEAL_LEN" >> "$CSV"

echo "[✓] Done. See: $RUN"

