#!/bin/bash
# Resume pair-compare in chunks, waking at fixed session-reset boundaries.
#
# Anchor schedule (Claude Code Max 5-hour rolling window):
#   t0 + 2h    (next session reset from script start)
#   t0 + 7h
#   t0 + 12h
#   t0 + 17h ...
#
# Override the initial 2h offset:
#   NEXT_RESET_OFFSET_SEC=7200 bash failure-taxonomy/run_batches.sh
#
# Survives Claude Code closure when launched via:
#   nohup bash failure-taxonomy/run_batches.sh > /tmp/pair_loop.log 2>&1 &

set -u

cd /Users/hudx/Desktop/tb-work
source venv312/bin/activate

CACHE=failure-taxonomy/outputs/_pair_cache
TARGET=528
BATCH_LIMIT=50
PARALLEL=2

NEXT_RESET_OFFSET_SEC=${NEXT_RESET_OFFSET_SEC:-$((2 * 3600))}   # how far away the *next* session reset is, in sec
SESSION_WINDOW_SEC=$((5 * 3600))                                # 5h rolling window

SCRIPT_START=$(date +%s)
SESSION_ANCHOR=$((SCRIPT_START + NEXT_RESET_OFFSET_SEC))
echo "loop start: $(date -r $SCRIPT_START)"
echo "first session reset anchor: $(date -r $SESSION_ANCHOR)  (then every ${SESSION_WINDOW_SEC}s)"

iter=0
while true; do
  iter=$((iter + 1))
  iter_start=$(date +%s)
  cached_before=$(ls "$CACHE" 2>/dev/null | wc -l | tr -d ' ')
  echo ""
  echo "=== iter $iter — cached before: $cached_before / $TARGET — $(date) ==="

  if [ "$cached_before" -ge "$TARGET" ]; then
    echo "target reached, exiting"
    break
  fi

  BATCH_LOG="/tmp/pair_batch_iter${iter}.log"
  BATCH_LIMIT=$BATCH_LIMIT PARALLEL=$PARALLEL python -u failure-taxonomy/04_pair_compare.py \
      > >(tee "$BATCH_LOG") 2>&1
  rc=$?

  cached_after=$(ls "$CACHE" 2>/dev/null | wc -l | tr -d ' ')
  new_this_iter=$((cached_after - cached_before))

  rate_limit_msgs=$(grep -ciE "rate.limit|usage.limit|quota|429|too.many.requests|RateLimitError" "$BATCH_LOG" 2>/dev/null | head -1)
  rate_limit_msgs=${rate_limit_msgs:-0}
  error_lines=$(grep -c "ERROR" "$BATCH_LOG" 2>/dev/null | head -1)
  error_lines=${error_lines:-0}

  echo "  iter $iter rc=$rc new=$new_this_iter cached_after=$cached_after rate_msgs=$rate_limit_msgs errors=$error_lines"

  if [ "$cached_after" -ge "$TARGET" ]; then
    echo "target reached after iter $iter, exiting"
    break
  fi

  # Find next session-reset boundary after now.
  now=$(date +%s)
  if [ "$now" -lt "$SESSION_ANCHOR" ]; then
    next_wake=$SESSION_ANCHOR
  else
    elapsed=$((now - SESSION_ANCHOR))
    k=$((elapsed / SESSION_WINDOW_SEC + 1))
    next_wake=$((SESSION_ANCHOR + k * SESSION_WINDOW_SEC))
  fi

  sleep_for=$((next_wake - now))
  if [ "$sleep_for" -lt 60 ]; then
    # boundary is right around now — wait a tiny buffer so reset takes effect
    sleep_for=60
    next_wake=$((now + 60))
  fi
  echo "  sleeping ${sleep_for}s — next iter at $(date -r $next_wake)"
  sleep "$sleep_for"
done

echo ""
echo "=== done $(date) ==="
