#!/usr/bin/env bash
# Plan 2 end-to-end pipeline: wait for embeds -> smoke -> pilot -> full sweep.
# Runs under nohup; each stage's exit code is checked before the next launches.
# Usage:
#   nohup ./skills\ retrieval/run_plan2_pipeline.sh > skills\ retrieval/runs/plan2_pipeline.log 2>&1 &
#   disown

set -u
set -o pipefail

cd "/anvil/projects/x-cis260386/william/procmem2skills/procmem2skills"

PY=/anvil/projects/x-cis260386/william/icy/envs/ground-r1/bin/python3.10
export PYTHONPATH="skills retrieval/src"

EMBEDS="skills retrieval/pools/tasks_gt_embeddings_full.npz"
LOG_DIR="skills retrieval/runs"
mkdir -p "$LOG_DIR"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# -------- Stage 0: wait for full embeddings --------
log "Stage 0: waiting for $EMBEDS ..."
while [ ! -f "$EMBEDS" ]; do sleep 15; done
log "Stage 0: embeds present."
"$PY" -c "
import numpy as np
d = np.load('$EMBEDS')
print({k: d[k].shape for k in d.files})
" || { log "embed inspection failed"; exit 1; }

# -------- Stage 1: smoke (single task, 2 API calls) --------
log "Stage 1: smoke test ..."
"$PY" -m skills_retrieval.run_m2 \
    --task-ids sb_000 --pool-sizes 5 --strategies random --seeds 0 \
    --label plan2-smoke 2>&1 | tee "$LOG_DIR/plan2-smoke.log"
SMOKE_EXIT=${PIPESTATUS[0]}
if [ "$SMOKE_EXIT" -ne 0 ]; then
    log "Stage 1 FAILED (exit $SMOKE_EXIT); aborting pipeline."
    exit 1
fi

# Sanity check the smoke summary
SMOKE_DIR=$(ls -dt "skills retrieval/runs/"*plan2-smoke* 2>/dev/null | head -1)
if [ -z "$SMOKE_DIR" ] || [ ! -s "$SMOKE_DIR/metrics/summary_by_condition.json" ]; then
    log "Stage 1: smoke summary missing; aborting."
    exit 1
fi
log "Stage 1 OK. Summary: $SMOKE_DIR/metrics/summary_by_condition.json"
"$PY" -c "
import json, sys
d = json.load(open('$SMOKE_DIR/metrics/summary_by_condition.json'))
r = d.get('random__n5', {}).get('awareness_recall5', {}).get('mean')
print('smoke random@n5 awareness_recall5 =', r)
sys.exit(0 if (r is not None and r >= 0.5) else 2)
" || { log "Stage 1: smoke recall check failed; aborting."; exit 1; }

# -------- Stage 2: pilot (9 tasks × 2 strategies × 3 N × 3 seeds = 324 calls) --------
log "Stage 2: pilot run ..."
"$PY" -m skills_retrieval.run_m2 \
    --task-ids sb_000 sb_002 sb_003 sb_004 sb_005 sb_006 sb_007 sb_008 sb_009 \
    --pool-sizes 5 50 200 \
    --strategies random hard_neg_semantic \
    --seeds 0 1 2 \
    --concurrency 4 \
    --label plan2-pilot 2>&1 | tee "$LOG_DIR/plan2-pilot.log"
PILOT_EXIT=${PIPESTATUS[0]}
if [ "$PILOT_EXIT" -ne 0 ]; then
    log "Stage 2 FAILED (exit $PILOT_EXIT); aborting."
    exit 1
fi
PILOT_DIR=$(ls -dt "skills retrieval/runs/"*plan2-pilot* 2>/dev/null | head -1)
log "Stage 2 OK. Pilot dir: $PILOT_DIR"

"$PY" -c "
import json, sys
d = json.load(open('$PILOT_DIR/metrics/summary_by_condition.json'))
r5 = d.get('random__n5', {}).get('awareness_recall5', {}).get('mean', 0)
r_hn = d.get('hard_neg_semantic__n200', {}).get('awareness_recall5', {}).get('mean', 1)
r_rnd200 = d.get('random__n200', {}).get('awareness_recall5', {}).get('mean', 0)
print(f'random@5={r5} hard_neg@200={r_hn} random@200={r_rnd200}')
# Loose sanity: random@5 recall high, hard_neg@200 < random@200 typically
sys.exit(0 if r5 >= 0.5 else 2)
" || { log "Stage 2 sanity check failed; aborting full sweep."; exit 1; }

# -------- Stage 3: full 88-task sweep --------
log "Stage 3: full sweep ..."
"$PY" -m skills_retrieval.run_m2 \
    --concurrency 4 \
    --label plan2-m2 2>&1 | tee "$LOG_DIR/plan2-m2.log"
FULL_EXIT=${PIPESTATUS[0]}
if [ "$FULL_EXIT" -ne 0 ]; then
    log "Stage 3 exited $FULL_EXIT (may be partial; resume with --skip-existing --out-dir)."
    exit "$FULL_EXIT"
fi
log "Stage 3 OK. Pipeline complete."
