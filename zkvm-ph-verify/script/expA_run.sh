#!/usr/bin/env bash
# expA_run.sh — Anchor/Verify 반복 실험 + 가스/지연/로그 수집 (Sepolia, MockRouter)
# 필요한 도구: cast, jq, python3, xxd, dd
set -euo pipefail

### ─────────────────────────────────────────────────────────────────────────────
### 0) 환경 변수 로드
###    ./env.sh 가 있으면 source, 없으면 아래 필수 ENV를 직접 채우세요.
### ─────────────────────────────────────────────────────────────────────────────
# 영수증 대기(초)
: "${RECEIPT_TIMEOUT:=600}"
: "${RECEIPT_INTERVAL:=2}"

# 가스 설정 (wei 단위)
: "${PRIORITY_GAS_PRICE_WEI=5000000000}"   # 5 gwei
: "${GAS_PRICE_WEI=9000000000}"            # 9 gwei
: "${REGISTER_GAS_LIMIT:=120000}"
: "${VERIFY_GAS_LIMIT:=300000}"

# 32바이트 hex (0x + 64hex) 검증
require_hex32() { # $1=VAR_NAME
  local name="$1"
  local val="${!name:-}"
  if [[ ! "$val" =~ ^0x[0-9a-fA-F]{64}$ ]]; then
    echo "ERROR: $name must be 0x + 64 hex chars; got '$val'" >&2
    exit 1
  fi
}
require_hex32 PID
require_hex32 PH

# 의존성 체크
need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found in PATH"; exit 1; }; }
need cast; need jq; need python3; need xxd; need dd

# 필수값 확인
missing=()
for v in RPC PK REG ROUTER PID PH SEAL_BIN; do
  [[ -z "${!v}" ]] && missing+=("$v")
done
if (( ${#missing[@]} > 0 )); then
  echo "ERROR: Missing env vars: ${missing[*]}"
  echo "  -> 준비: export RPC=... PK=... REG=... ROUTER=... PID=... PH=... SEAL_BIN=..."
  exit 1
fi
if [[ ! -f "$SEAL_BIN" ]]; then
  echo "ERROR: SEAL_BIN file not found: $SEAL_BIN"
  exit 1
fi

### ─────────────────────────────────────────────────────────────────────────────
### 1) 유틸 함수
### ─────────────────────────────────────────────────────────────────────────────
now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }
now_ms() {
python3 - <<'PY'
import time
print(int(time.time()*1000))
PY
}

wait_receipt() { # $1=txhash
  local tx="$1"
  local timeout="$RECEIPT_TIMEOUT"
  local interval="$RECEIPT_INTERVAL"
  local start
  start=$(date +%s)
  while true; do
    # ✅ RPC 지정 필수
    if cast receipt "$tx" --rpc-url "$RPC" --json > /tmp/rcp.json 2>/dev/null; then
      return 0
    fi
    local now
    now=$(date +%s)
    if (( now - start >= timeout )); then
      return 1
    fi
    sleep "$interval"
  done
}

# /tmp/rcp.json → status, gasUsed, effectiveGasPrice, blockNumber
grab_core_fields_from_receipt() {
  jq -r '[.status,.gasUsed,.effectiveGasPrice,.blockNumber] | @tsv' /tmp/rcp.json
}

# /tmp/rcp.json → baseFeePerGas (영수증 blockNumber 사용)
# /tmp/rcp.json → baseFeePerGas (영수증 blockNumber 사용; 비어있으면 빈 문자열 반환)
get_basefee_wei_from_receipt() {
  local blockHex
  blockHex="$(jq -r 'try .blockNumber // empty' /tmp/rcp.json)"
  if [[ -z "$blockHex" || "$blockHex" == "null" ]]; then
    echo ""
    return 0
  fi

  local blockDec
  blockDec="$(python3 - "$blockHex" <<'PY'
import sys
s=sys.argv[1].strip()
try:
    n = int(s,16) if s.startswith(('0x','0X')) else int(s)
    print(n)
except Exception:
    print("")
PY
)"
  if [[ -z "$blockDec" ]]; then
    echo ""
    return 0
  fi

  # ✅ RPC 지정 필수
  cast block "$blockDec" --rpc-url "$RPC" --json | jq -r 'try .baseFeePerGas // empty'
}

# 로그 파싱(라우터/레지스트리) — jq 1.6 호환 위해 python 사용
parse_logs_python() { # 출력 TSV: routerLog registryLog routerImageId routerJournalDigest routerSealLen registryProgramId registryPH
python3 - "$ROUTER" "$REG" <<'PY'
import sys,json
router = sys.argv[1].lower()
reg    = sys.argv[2].lower()
with open("/tmp/rcp.json","r") as f:
    r = json.load(f)

def words(hexdata):
    if not hexdata: return []
    h = hexdata[2:] if hexdata.startswith("0x") else hexdata
    return ["0x"+h[i:i+64] for i in range(0,len(h),64)]

logs = r.get("logs",[])
rlog = [L for L in logs if L.get("address","").lower()==router]
glog = [L for L in logs if L.get("address","").lower()==reg]

routerLog = 1 if rlog else 0
registryLog = 1 if glog else 0

rid = jdg = slen = rpid = rph = ""

if rlog:
    w = words(rlog[0].get("data","0x"))
    # 예상 레이아웃: [ offset, imageId, journalDigest, seal_len, seal_words... ]
    if len(w) >= 4:
        rid = w[1]
        jdg = w[2]
        slen = w[3]
if glog:
    w = words(glog[0].get("data","0x"))
    if len(w) >= 2:
        rpid = w[0]
        rph  = w[1]

print(f"{routerLog}\t{registryLog}\t{rid}\t{jdg}\t{slen}\t{rpid}\t{rph}")
PY
}

# 헥스 도우미들
hex_of_file() { xxd -p -c 999999 "$1" | tr -d '\n'; }
flip_hex32_bit0() { # $1 = 0x + 64 hex
python3 - "$1" <<'PY'
import sys,re
s=(sys.argv[1] if len(sys.argv)>1 else "").strip()
if not re.fullmatch(r"0x[0-9a-fA-F]{64}", s):
    raise SystemExit(f"flip_hex32_bit0: invalid hex32 input: {s!r}")
x=int(s[2:],16)^1
print("0x"+format(x,'064x'))
PY
}

make_seal_bad() { # $1=src $2=dst [$3=offset=41]
  local src="$1"; local dst="$2"; local off="${3:-41}"
  cp "$src" "$dst"
  printf '\x00' | dd of="$dst" bs=1 seek="$off" count=1 conv=notrunc status=none
}

TX_OPTS=( --priority-gas-price "$PRIORITY_GAS_PRICE_WEI" --gas-price "$GAS_PRICE_WEI" )

send_register() { # $1=pid $2=ph
  cast send "$REG" 'register(bytes32,bytes32)' "$1" "$2" \
    --gas-limit "$REGISTER_GAS_LIMIT" "${TX_OPTS[@]}" \
    --rpc-url "$RPC" --private-key "$PK" --json | jq -r .transactionHash
}

send_verify() { # $1=pid $2=ph $3=seal_hex $4=journal_hex
  cast send "$REG" 'verify(bytes32,bytes32,bytes,bytes)' "$1" "$2" "0x$3" "$4" \
    --gas-limit "$VERIFY_GAS_LIMIT" "${TX_OPTS[@]}" \
    --rpc-url "$RPC" --private-key "$PK" --json 2>/tmp/send_err.json | jq -r .transactionHash
}

### ─────────────────────────────────────────────────────────────────────────────
### 2) 출력 경로 & 헤더
### ─────────────────────────────────────────────────────────────────────────────
OUTDIR="runs/expA_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/results.csv"
echo "ts,case,trial,tx,status,statusText,gasUsed,effectiveGasPriceWei,costWei,baseFeeWei,priorityFeeWei,latencyMs,routerLog,registryLog,routerImageId,routerJournalDigest,routerSealLen,registryProgramId,registryPH,mutateField,mutatePos,programId,ph,router,registry,sealBytes,notes" > "$CSV"

### ─────────────────────────────────────────────────────────────────────────────
### 3) 준비물: seal, 변조값
### ─────────────────────────────────────────────────────────────────────────────
SEAL_HEX="$(hex_of_file "$SEAL_BIN")"
SEAL_BAD_BIN="$OUTDIR/seal_bad.bin"; make_seal_bad "$SEAL_BIN" "$SEAL_BAD_BIN" 41
SEAL_BAD_HEX="$(hex_of_file "$SEAL_BAD_BIN")"
PH_ALT="$(flip_hex32_bit0 "$PH")"
PID_ALT="$(flip_hex32_bit0 "$PID")"

### ─────────────────────────────────────────────────────────────────────────────
### 4) 1회 앵커(register)
### ─────────────────────────────────────────────────────────────────────────────
echo "[*] Register anchor once…"
TXR="$(send_register "$PID" "$PH" || true)"
if [[ -z "${TXR:-}" || "$TXR" == "null" ]]; then
  echo "WARN: register tx not returned; check keys/funds/RPC"
else
  if ! wait_receipt "$TXR"; then
    echo "WARN: register receipt timeout for $TXR"
  fi
fi

# 반복 실행 함수 (verify 케이스)
# 사용법: run_case "<caseName>" "<seal_hex>" "<pid>" "<ph>" "<journal>" "<mutateField>" "<mutatePos>"
run_case() {
  local case="$1" seal_hex="$2" pid="$3" ph="$4" journal="$5" mutateField="$6" mutatePos="$7"

  for ((i=1;i<=200;i++)); do
    # 100~300ms 랜덤 간격으로 호출 (nonce/네트워크 충돌 완화)
    sleep 0.$(( (RANDOM%3)+1 ))

    local ts t0 t1 tx status statusText gas egp baseFee priority cost latency notes
    local rlog glog rid jdg slen rpid rph

    ts="$(now_iso)"
    t0="$(now_ms)"

    # verify 트랜잭션 전송
    tx="$(send_verify "$pid" "$ph" "$seal_hex" "$journal" || true)"

    if [[ -z "${tx:-}" || "$tx" == "null" ]]; then
      # 전송 실패 (RPC/키/네트워크 오류)
      status="infra_error"; statusText="infra_error"
      gas=""; egp=""; cost=""; baseFee=""; priority=""; latency=""
      rlog=0; glog=0; rid=""; jdg=""; slen=""; rpid=""; rph=""
      notes="$(tr '\n' ' ' </tmp/send_err.json 2>/dev/null || true)"

    else
      # 영수증 대기 (cast wait 대체)
      if ! wait_receipt "$tx"; then
        status="infra_timeout"; statusText="infra_timeout"
        gas=""; egp=""; cost=""; baseFee=""; priority=""
        rlog=0; glog=0; rid=""; jdg=""; slen=""; rpid=""; rph=""
        latency=$(( $(now_ms) - t0 ))
        notes=""

      else
        # 영수증 확보 → 기본 필드 파싱
        t1="$(now_ms)"
        read status gas egp _ < <(grab_core_fields_from_receipt)
        if [[ "$status" == "0x1" || "$status" == "1" ]]; then
          statusText="success"
        else
          statusText="reverted"
        fi

        # 비용 계산 (안전 가드)
        if [[ -n "${gas:-}" && "$gas" != "null" && -n "${egp:-}" && "$egp" != "null" ]]; then
          cost="$(python3 - <<PY
g="$gas"; p="$egp"
try:
    print(int(g,0)*int(p,0))
except Exception:
    print("")
PY
)"
        else
          cost=""
        fi

        # baseFee / priority 계산 (blockNumber가 비어있을 수 있어 가드)
        baseFee="$(get_basefee_wei_from_receipt || echo "")"
        if [[ -n "${egp:-}" && "$egp" != "null" && -n "${baseFee:-}" && "$baseFee" != "null" && -n "$baseFee" ]]; then
          priority="$(python3 - <<PY
e="$egp"; b="$baseFee"
try:
    print(int(e,0)-int(b,0))
except Exception:
    print("")
PY
)"
        else
          priority=""
        fi

        latency=$(( t1 - t0 ))

        # 라우터/레지스트리 로그 파싱 (없으면 0/빈값)
        read rlog glog rid jdg slen rpid rph < <(parse_logs_python)
        notes=""
      fi
    fi

    # CSV 라인 기록 (헤더 순서에 맞춤)
    echo "$ts,$case,$i,$tx,$status,$statusText,$gas,$egp,$cost,$baseFee,${priority:-},$latency,$rlog,$glog,$rid,$jdg,$slen,$rpid,$rph,$mutateField,$mutatePos,$PID,$PH,$ROUTER,$REG,260,$notes" >> "$CSV"
  done
}

### ─────────────────────────────────────────────────────────────────────────────
### 6) 케이스 실행
###     A1: 정상 / A2: PH 변조 / A3: journal 변조 / A4: PID 변조 / A5: seal 변조(Mock 한계)
### ─────────────────────────────────────────────────────────────────────────────
echo "[*] Run cases (N=200 per case)…"
run_case "A1_ok"          "$SEAL_HEX"     "$PID"     "$PH"     "$PH"       "none"    ""
run_case "A2_ph_mut"      "$SEAL_HEX"     "$PID"     "$PH_ALT" "$PH_ALT"   "PH"      "n/a"
run_case "A3_journal_mut" "$SEAL_HEX"     "$PID"     "$PH"     "$PH_ALT"   "journal" "n/a"
run_case "A4_pid_mut"     "$SEAL_HEX"     "$PID_ALT" "$PH"     "$PH"       "PID"     "n/a"
run_case "A5_seal_mut"    "$SEAL_BAD_HEX" "$PID"     "$PH"     "$PH"       "seal"    "41"

echo "[✓] DONE -> $CSV"
echo "    예) 요약: python3 script/expA_summary.py $CSV"

