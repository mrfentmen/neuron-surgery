#!/usr/bin/env bash
# serve.sh — the one command. Local agentic coding model that stays cool.
#
#   ./serve.sh              # default: 9B, 64K context, cool profile
#   ./serve.sh long         # 9B, 262K context
#   ./serve.sh big          # 35B-A3B, 64K — better model, same-ish heat
#   ./serve.sh big long     # 35B-A3B, 262K — best quality, watch memory
#
# Heat control is NOT a flag on the model, it is how hard you lean on it.
# This script defaults to settings that keep the GPU at a sustainable duty cycle
# rather than max throughput. See THERMAL.md.

set -euo pipefail
cd "$(dirname "$0")"

PORT=${PORT:-8080}
PROFILE="cool"
CTX=65536

for arg in "$@"; do
  case "$arg" in
    long)  CTX=262144 ;;
    big)   PROFILE="big" ;;
    cool)  CTX=65536 ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

if [ "$PROFILE" = "big" ]; then
  MODEL="models/qwen35-35ba3b/Qwen3.5-35B-A3B-Q4_K_M.gguf"
  THREADS=6
else
  MODEL="models/qwen35/Qwen3.5-9B-Q4_K_M.gguf"
  THREADS=8
fi

[ -f "$MODEL" ] || { echo "missing model: $MODEL"; echo "download it first — see MODEL-PICK.md"; exit 1; }

# KV quantized to q8: halves the cache, costs ~nothing in quality, and keeps
# a 262K run inside the wired ceiling. q4 only if you go to 1M.
KV="q8_0"
if [ "$CTX" -ge 524288 ]; then KV="q4_0"; fi

echo "────────────────────────────────────────────────"
echo " model   $(basename "$MODEL")"
echo " context $CTX tokens   KV cache $KV"
echo " port    http://127.0.0.1:$PORT/v1"
echo
echo " thinking is OFF by default — send"
echo '   "chat_template_kwargs": {"enable_thinking": false}'
echo " or you will get empty content and burn your budget on reasoning."
echo "────────────────────────────────────────────────"
echo

exec llama-server \
  -m "$MODEL" \
  -c "$CTX" \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  --flash-attn on \
  --jinja \
  -t "$THREADS" \
  -ngl 99 \
  --port "$PORT"
