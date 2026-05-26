#!/usr/bin/env bash
# Phase B: strip rows polluted by UniAPI Bedrock "invalid beta flag" errors,
# then runner --resume will re-do the (task, pool, noise, seed) cells.
#
# Safe to re-run. Original jsonl backed up before edit.
set -euo pipefail
cd "$(dirname "$0")"

OUT="results/smoke20/smoke20_n5.jsonl"
BACKUP="${OUT}.beta_pollution_$(date +%Y%m%dT%H%M%S).bak"

if [ ! -f "$OUT" ]; then
    echo "ERROR: $OUT not found"
    exit 1
fi

# Backup
cp "$OUT" "$BACKUP"
echo "Backed up to: $BACKUP"

python3 << EOF
import json
src = "$OUT"
dst = src + ".filtered"
removed = []
kept = 0
with open(src) as f, open(dst, 'w') as o:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        # Strip rows where the agent died on UniAPI Bedrock validation
        if 'invalid beta flag' in (r.get('agent_stdout_tail') or '').lower():
            removed.append(r.get('trial_id'))
            continue
        o.write(line)
        kept += 1
print(f"Removed {len(removed)} polluted rows, kept {kept} clean rows")
import os
os.rename(dst, src)
print(f"In-place rewrite done. Backup at $BACKUP")
EOF

echo
echo "Now restart container to pick up the cleaned jsonl + --resume:"
echo "  sg docker -c 'docker stop smoke20_next'"
echo "  bash launch_smoke20_next.sh"
echo
echo "The runner will see those cells as undone and re-attempt them."
