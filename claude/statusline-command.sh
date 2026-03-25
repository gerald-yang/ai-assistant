#!/bin/bash
# Read all of stdin into a variable
input=$(cat)

# --- Who am I and launch directory ---
USER_NAME=$(whoami)
LAUNCH_DIR=$(echo "$input" | jq -r '.workspace.project_dir // .cwd')

# --- Model ---
MODEL=$(echo "$input" | jq -r '.model.display_name')

# --- Context progress bar ---
PCT_RAW=$(echo "$input" | jq -r '.context_window.used_percentage // empty')
if [ -n "$PCT_RAW" ]; then
  PCT=$(printf "%.0f" "$PCT_RAW")
else
  PCT=0
fi
BAR_WIDTH=10
FILLED=$((PCT * BAR_WIDTH / 100))
EMPTY=$((BAR_WIDTH - FILLED))
BAR=""
[ "$FILLED" -gt 0 ] && printf -v FILL "%${FILLED}s" && BAR="${FILL// /▓}"
[ "$EMPTY" -gt 0 ] && printf -v PAD "%${EMPTY}s" && BAR="${BAR}${PAD// /░}"
CTX_BAR="[${BAR}] ${PCT}%"

# --- Rate limit progress bar helper ---
make_bar() {
  local pct=$1
  local width=10
  local filled=$(( pct * width / 100 ))
  local empty=$(( width - filled ))
  local bar=""
  [ "$filled" -gt 0 ] && printf -v f "%${filled}s" && bar="${f// /▓}"
  [ "$empty" -gt 0 ] && printf -v e "%${empty}s" && bar="${bar}${e// /░}"
  echo "[${bar}] ${pct}%"
}

# --- Rate limits ---
FIVE_RAW=$(echo "$input" | jq -r '.rate_limits.five_hour.used_percentage // empty')
WEEK_RAW=$(echo "$input" | jq -r '.rate_limits.seven_day.used_percentage // empty')
RATE_PARTS=""
if [ -n "$FIVE_RAW" ]; then
  FIVE=$(printf "%.0f" "$FIVE_RAW")
  FIVE_BAR=$(make_bar "$FIVE")
  RATE_PARTS="5h:${FIVE_BAR}"
fi
if [ -n "$WEEK_RAW" ]; then
  WEEK=$(printf "%.0f" "$WEEK_RAW")
  WEEK_BAR=$(make_bar "$WEEK")
  [ -n "$RATE_PARTS" ] && RATE_PARTS="${RATE_PARTS} "
  RATE_PARTS="${RATE_PARTS}7d:${WEEK_BAR}"
fi

# --- Session cost ---
INPUT_TOKENS=$(echo "$input" | jq -r '.context_window.total_input_tokens // 0')
OUTPUT_TOKENS=$(echo "$input" | jq -r '.context_window.total_output_tokens // 0')
# Pricing for claude-sonnet-4-6: $3/M input, $15/M output (standard rates)
COST=$(awk "BEGIN { printf \"%.4f\", ($INPUT_TOKENS / 1000000 * 3) + ($OUTPUT_TOKENS / 1000000 * 15) }")
COST_STR="\$${COST}"

# --- Assemble output ---
LINE="${USER_NAME} 📁 ${LAUNCH_DIR} | ${MODEL} | ctx:${CTX_BAR}"
[ -n "$RATE_PARTS" ] && LINE="${LINE} | ${RATE_PARTS}"
LINE="${LINE} | 💰 ${COST_STR}"

printf "%s" "$LINE"
