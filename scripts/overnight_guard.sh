#!/bin/bash
set -euo pipefail

BOT_DIR="${PM_BOT_DIR:-/opt/polymarket-bot}"
API_URL="${PM_BOT_URL:-http://127.0.0.1:8075}"
DATA_DIR="$BOT_DIR/data"
STATE_FILE="$DATA_DIR/overnight_guard_state.json"
REPORT_FILE="$DATA_DIR/overnight_guard_report.log"
DISABLE_SNIPER_FLAG="$DATA_DIR/disable_sniper.flag"
DISABLE_CERTAINTY_FLAG="$DATA_DIR/disable_certainty.flag"

SNIPER_CMD=(/opt/polymarket-bot/venv/bin/python3 -u /opt/polymarket-bot/run_sniper.py 604800 151.248)
CERTAINTY_CMD=(/opt/polymarket-bot/venv/bin/python3 -u /opt/polymarket-bot/run_certainty_sniper.py 604800 50.0)
API_CMD=(/opt/polymarket-bot/venv/bin/python3 -u /opt/polymarket-bot/venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8075)

MAX_RESTARTS_PER_SERVICE=3
SNIPER_MAX_REALIZED_LOSS=15.0

mkdir -p "$DATA_DIR"
ts="$(date -u +"%Y-%m-%d %H:%M:%S UTC")"

read_state() {
  if [[ -f "$STATE_FILE" ]]; then
    cat "$STATE_FILE"
  else
    echo '{"date":"","restarts":{"api":0,"sniper":0,"certainty":0},"disabled":{"sniper":false,"certainty":false}}'
  fi
}

write_state() {
  printf '%s\n' "$1" > "$STATE_FILE"
}

state_json="$(read_state)"
today="$(date -u +%F)"

state_json="$(STATE_JSON="$state_json" venv/bin/python - "$today" <<'PY'
import json, os, sys
state = json.loads(os.environ["STATE_JSON"])
today = sys.argv[1]
if state.get("date") != today:
    state = {
        "date": today,
        "restarts": {"api": 0, "sniper": 0, "certainty": 0},
        "disabled": {"sniper": False, "certainty": False},
    }
print(json.dumps(state))
PY
)"

json_get() {
  local expr="$1"
  STATE_JSON="$state_json" venv/bin/python - "$expr" <<'PY'
import json, os, sys
obj = json.loads(os.environ["STATE_JSON"])
expr = sys.argv[1].split(".")
cur = obj
for part in expr:
    cur = cur[part]
print(json.dumps(cur))
PY
}

json_set() {
  local expr="$1"
  local value="$2"
  state_json="$(STATE_JSON="$state_json" venv/bin/python - "$expr" "$value" <<'PY'
import json, os, sys
obj = json.loads(os.environ["STATE_JSON"])
path = sys.argv[1].split(".")
value = json.loads(sys.argv[2])
cur = obj
for part in path[:-1]:
    cur = cur[part]
cur[path[-1]] = value
print(json.dumps(obj))
PY
)"
}

restart_count() {
  local svc="$1"
  json_get "restarts.$svc" | tr -d '"'
}

disabled_flag() {
  local svc="$1"
  local state_flag
  state_flag="$(json_get "disabled.$svc" | tr -d '"')"
  if [[ "$svc" == "sniper" && -f "$DISABLE_SNIPER_FLAG" ]]; then
    echo "true"
    return
  fi
  if [[ "$svc" == "certainty" && -f "$DISABLE_CERTAINTY_FLAG" ]]; then
    echo "true"
    return
  fi
  echo "$state_flag"
}

increment_restart() {
  local svc="$1"
  local current
  current="$(restart_count "$svc")"
  json_set "restarts.$svc" "$((current + 1))"
}

disable_service() {
  local svc="$1"
  json_set "disabled.$svc" "true"
}

log_line() {
  printf '[%s] %s\n' "$ts" "$1" >> "$REPORT_FILE"
}

start_api() {
  setsid -f "${API_CMD[@]}" >> "$BOT_DIR/uvicorn.log" 2>&1
}

start_sniper() {
  setsid -f "${SNIPER_CMD[@]}" >> "$BOT_DIR/sniper_output.log" 2>&1
}

start_certainty() {
  setsid -f "${CERTAINTY_CMD[@]}" >> "$BOT_DIR/certainty_sniper_output.log" 2>&1
}

service_running() {
  local pattern="$1"
  pgrep -af "$pattern" >/dev/null 2>&1
}

api_json='{}'
if service_running "uvicorn src.main:app --host 0.0.0.0 --port 8075"; then
  api_json="$(venv/bin/python - <<'PY'
import json, urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:8075/api/all-status', timeout=5) as r:
        print(json.dumps(json.load(r)))
except Exception:
    print("{}")
PY
)"
fi

sniper_realized_pnl="$(API_JSON="$api_json" venv/bin/python - <<'PY'
import json, os
try:
    data = json.loads(os.environ["API_JSON"])
    print(float(data.get("sniper", {}).get("scorecard", {}).get("live", {}).get("realized_pnl", 0.0)))
except Exception:
    print(0.0)
PY
)"

sniper_live_count="$(API_JSON="$api_json" venv/bin/python - <<'PY'
import json, os
try:
    data = json.loads(os.environ["API_JSON"])
    print(int(data.get("sniper", {}).get("scorecard", {}).get("live", {}).get("count", 0)))
except Exception:
    print(0)
PY
)"

if (( $(printf '%.0f' "${sniper_realized_pnl#-}") >= 0 )); then :; fi

if [[ "$(disabled_flag sniper)" != "true" && "$sniper_live_count" -ge 3 ]]; then
  if venv/bin/python - "$sniper_realized_pnl" "$SNIPER_MAX_REALIZED_LOSS" <<'PY'
import sys
pnl = float(sys.argv[1]); limit = float(sys.argv[2])
raise SystemExit(0 if pnl <= -limit else 1)
PY
  then
    pkill -f "/opt/polymarket-bot/run_sniper.py 604800 151.248" || true
    disable_service sniper
    log_line "Disabled sniper after realized live loss hit ${sniper_realized_pnl}"
  fi
fi

if ! service_running "uvicorn src.main:app --host 0.0.0.0 --port 8075"; then
  if [[ "$(restart_count api)" -lt "$MAX_RESTARTS_PER_SERVICE" ]]; then
    start_api
    increment_restart api
    log_line "Restarted api"
  else
    log_line "API down but restart budget exhausted"
  fi
fi

if [[ "$(disabled_flag sniper)" != "true" ]] && ! service_running "/opt/polymarket-bot/run_sniper.py 604800 151.248"; then
  if [[ "$(restart_count sniper)" -lt "$MAX_RESTARTS_PER_SERVICE" ]]; then
    start_sniper
    increment_restart sniper
    log_line "Restarted sniper"
  else
    log_line "Sniper down but restart budget exhausted"
  fi
fi

if [[ "$(disabled_flag certainty)" != "true" ]] && ! service_running "/opt/polymarket-bot/run_certainty_sniper.py 604800 50.0"; then
  if [[ "$(restart_count certainty)" -lt "$MAX_RESTARTS_PER_SERVICE" ]]; then
    start_certainty
    increment_restart certainty
    log_line "Restarted certainty"
  else
    log_line "Certainty down but restart budget exhausted"
  fi
fi

summary="$(API_JSON="$api_json" venv/bin/python - <<'PY'
import json, os
try:
    data = json.loads(os.environ["API_JSON"])
    sniper = data.get("sniper", {})
    certainty = data.get("certainty_sniper", {})
    print(
        f"sniper_pnl={sniper.get('pnl', 0)} "
        f"sniper_live_trades={sniper.get('scorecard', {}).get('live', {}).get('count', 0)} "
        f"certainty_open={certainty.get('open', 0)}"
    )
except Exception:
    print("api_unavailable")
PY
)"
log_line "$summary"

write_state "$state_json"
