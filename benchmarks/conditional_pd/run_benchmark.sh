#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# A/B: baseline PD vs conditional PD on 2 GPUs (Mooncake trace, 50 requests).
# Prereqs: etcd+NATS (docker compose -f dev/docker-compose.yml up -d), aiperf, model cached.
#
# Usage: ./run_benchmark.sh
# Env:   MODEL, NUM_REQUESTS, CONCURRENCY, DYN_CONDITIONAL_PD (set by script per arm)

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
DISAGG="$REPO_ROOT/examples/backends/vllm/launch/disagg.sh"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
NUM_REQUESTS="${NUM_REQUESTS:-50}"
CONCURRENCY="${CONCURRENCY:-8}"
WARMUP="${WARMUP_REQUESTS:-8}"
SEED="${SEED:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
MOONCAKE_URL="${MOONCAKE_URL:-https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl}"

# Pin ONE HF cache dir so the pre-download and both workers agree on location.
# Without this the pre-download can land in HF_HOME while the Dynamo workers
# use ~/.cache/huggingface/hub, so both workers still race and hit a lock.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

ensure_model_cached() {
    find "$HF_HUB_CACHE" -name "*.lock" -delete 2>/dev/null || true
    if python3 - <<'PY' 2>/dev/null; then
import os
from huggingface_hub import snapshot_download
print("Model cached:", snapshot_download(os.environ["MODEL"], local_files_only=True))
PY
        return 0
    fi
    echo "Downloading $MODEL into $HF_HUB_CACHE (once, before workers start) ..."
    python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["MODEL"])
print("Download complete.")
PY
}

mkdir -p "$DATA_DIR" "$RESULTS_DIR"
ensure_model_cached
TRACE="$DATA_DIR/mooncake_${NUM_REQUESTS}.jsonl"
if [[ ! -f "$TRACE" ]]; then
    SRC="$DATA_DIR/mooncake_trace.jsonl"
    [[ -f "$SRC" ]] || curl -fsSL "$MOONCAKE_URL" -o "$SRC"
    head -n "$NUM_REQUESTS" "$SRC" > "$TRACE"
fi

wait_ready() {
    local deadline=$((SECONDS + 300))
    until curl -fsS http://localhost:8081/health >/dev/null 2>&1; do
        (( SECONDS >= deadline )) && { echo "health check timed out" >&2; return 1; }
        sleep 5
    done
}

run_aiperf() {
    aiperf profile \
        --model "$MODEL" --tokenizer "$MODEL" --url http://localhost:8000 \
        --input-file "$TRACE" --custom-dataset-type mooncake_trace \
        --fixed-schedule --fixed-schedule-auto-offset \
        --prompt-input-tokens-block-size "$BLOCK_SIZE" --random-seed "$SEED" \
        --concurrency "$CONCURRENCY" --warmup-request-count "$WARMUP" \
        --endpoint-type chat --streaming --extra-inputs ignore_eos:true --no-gpu-telemetry \
        -H "Authorization: Bearer NOT USED" -H "Accept: text/event-stream" \
        --artifact-dir "$1"
}

run_arm() {
    local arm_name="$1"
    local pd="$2"
    local out="$RESULTS_DIR/$arm_name"
    echo "=== $arm_name (DYN_CONDITIONAL_PD=$pd) ==="
    export MODEL DYN_ROUTER_MODE=kv DYN_CONDITIONAL_PD="$pd"
    bash "$DISAGG" &
    local pid=$!
    trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' RETURN
    wait_ready
    run_aiperf "$out"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 10
    trap - RETURN
}

echo "Model=$MODEL  trace=$TRACE  requests=$NUM_REQUESTS  concurrency=$CONCURRENCY"
run_arm baseline 0
run_arm conditional_pd 1

python3 - "$RESULTS_DIR/baseline" "$RESULTS_DIR/conditional_pd" <<'PY'
import json, sys
from pathlib import Path

def load(path):
    f = next(Path(path).rglob("profile_export_aiperf.json"))
    return json.loads(f.read_text())

def avg(d, k):
    v = d.get(k) or {}
    return v.get("avg") if isinstance(v, dict) else None

data = {"baseline": load(sys.argv[1]), "conditional": load(sys.argv[2])}

def row(name, key):
    b, c = avg(data["baseline"], key), avg(data["conditional"], key)
    delta = f"{((c - b) / b * 100):+.1f}%" if b and c else "n/a"
    print(f"{name:28} {b or 0:10.2f} {c or 0:10.2f} {delta:>8}")

print("\nConditional PD vs baseline (lower TTFT/latency is better)")
print(f"{'Metric':28} {'Baseline':>10} {'Cond PD':>10} {'Delta':>8}")
row("TTFT (ms)", "time_to_first_token")
row("Request latency (ms)", "request_latency")
row("Throughput (req/s)", "request_throughput")
PY

echo "Artifacts: $RESULTS_DIR/{baseline,conditional_pd}"
