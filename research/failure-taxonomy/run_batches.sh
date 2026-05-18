#!/bin/bash
# Resume pair-compare in chunks; auto-detect usage-limit hits.
#
# Wait strategy:
#   - Normal between-batch sleep: 2 hours (covers 5-hour session window refresh)
#   - If batch hits usage limit (most calls failed): 7 hours (weekly safety margin)
#
# Survives terminal/Claude Code closure when launched via:
#   nohup bash failure-taxonomy/run_batches.sh > /tmp/pair_loop.log 2>&1 &

set -u

cd /Users/hudx/Desktop/tb-work
source venv312/bin/activate

CACHE=failure-taxonomy/outputs/_pair_cache
TARGET=528
BATCH_LIMIT=50
PARALLEL=2

NORMAL_WAIT=$((2 * 60 * 60))         # 2 hours
USAGE_LIMIT_WAIT=$((7 * 60 * 60))    # 7 hours

iter=0
while true; do
  iter=$((iter + 1))
  cached_before=$(ls "$CACHE" 2>/dev/null | wc -l | tr -d ' ')
  echo ""
  echo "=== iter $iter — cached before: $cached_before / $TARGET — $(date) ==="

  if [ "$cached_before" -ge "$TARGET" ]; then
    echo "target reached, exiting"
    break
  fi

  # Capture batch output to a temp file so we can scan it after
  BATCH_LOG="/tmp/pair_batch_iter${iter}.log"
  BATCH_LIMIT=$BATCH_LIMIT PARALLEL=$PARALLEL python -u failure-taxonomy/04_pair_compare.py \
      > >(tee "$BATCH_LOG") 2>&1
  rc=$?

  cached_after=$(ls "$CACHE" 2>/dev/null | wc -l | tr -d ' ')
  new_this_iter=$((cached_after - cached_before))

  # Usage-limit detection signals
  rate_limit_msgs=$(grep -ciE "rate.limit|usage.limit|quota|429|too.many.requests|RateLimitError" "$BATCH_LOG" 2>/dev/null | head -1)
  rate_limit_msgs=${rate_limit_msgs:-0}
  error_lines=$(grep -c "ERROR" "$BATCH_LOG" 2>/dev/null | head -1)
  error_lines=${error_lines:-0}

  echo "  iter $iter rc=$rc new_triples=$new_this_iter (cached_after=$cached_after)"
  echo "  rate-limit signals: $rate_limit_msgs  ERROR lines: $error_lines"

  if [ "$cached_after" -ge "$TARGET" ]; then
    echo "target reached after iter $iter, exiting"
    break
  fi

  # Decide wait length
  if [ "$rate_limit_msgs" -gt 0 ] || [ "$new_this_iter" -lt 10 ]; then
    wait_for=$USAGE_LIMIT_WAIT
    reason="usage-limit detected (rate_msgs=$rate_limit_msgs, new=$new_this_iter)"
  else
    wait_for=$NORMAL_WAIT
    reason="normal between-batch wait"
  fi

  next_at=$(date -v+${wait_for}S 2>/dev/null || date -d "+${wait_for}s" 2>/dev/null)
  echo "  [$reason] sleeping ${wait_for}s — next iter at $next_at"
  sleep "$wait_for"
done

echo ""
echo "=== done $(date) ==="
