#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"

echo "[*] Removing previous artifacts under $ROOT"
cd "$ROOT"

# 안전하게 우리 쪽 산출물만 정리
rm -rf runs \
       router.out.json reg.out.json \
       reg_register.tx.json reg_register.rcp.json \
       reg_verify.tx.json   reg_verify.rcp.json \
       ev.*.json \
       addresses.txt addresses.json export.env 2>/dev/null || true

# Foundry broadcast 로그(원하면 주석 처리)
rm -rf broadcast 2>/dev/null || true

echo "[✓] Cleaned."

