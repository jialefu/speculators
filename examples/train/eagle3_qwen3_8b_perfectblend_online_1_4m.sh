#!/bin/bash
# Online Eagle3 Training Script (Qwen3-8B + open-perfectblend)
#
# Step 1 only for now: on-policy response regeneration. Data preparation,
# vLLM hidden-states server, and training will be added once the regenerated
# data is in place.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/eagle3_qwen3_8b_perfectblend_online_1_4m.sh
#
# For a detailed walkthrough, see
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/response_regeneration/

### Example E2E run for Qwen3-8B on open-perfectblend (1.4M rows) ###

set -euo pipefail

# Optional comment tag appended to the wandb run name, e.g.:
#   bash examples/train/eagle3_qwen3_8b_perfectblend_online_1_4m.sh -c muon-lr-2e-4
COMMENT=""
while getopts "c:" opt; do
    case $opt in
        c) COMMENT="$OPTARG" ;;
        *) echo "Usage: $0 [-c comment]" >&2; exit 1 ;;
    esac
done

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="open-perfectblend"       # sharegpt, ultrachat, open-perfectblend, or path to custom data

# The regenerated JSONL must live OUTSIDE the training output dir:
# prepare_data.py --overwrite refuses to delete non-artifact files.
# Persistent storage (moonfs) survives reboots; /dev/shm does not.
PERSIST="/mnt/moonfs/fujiale-ksyun"
REGEN_DATA="$PERSIST/dataset/spec-regen-data/${DATASET}_$(basename "$MODEL").jsonl"
GPUS="0,1,2,3,4,5,6,7"          # GPUs for the regeneration vLLM server
# Throughput scales with replicas, not tensor parallelism. Qwen3-8B bf16 is
# ~16GB, so one replica per GPU (TP=1, DP=8) leaves huge KV-cache headroom on
# 143GB H200s and scales ~linearly. TP_SIZE x DP_SIZE must equal #GPUs.
TP_SIZE=1                       # Tensor parallel size per vLLM instance
DP_SIZE=8                       # Data parallel replicas behind the one endpoint
CONCURRENCY=2048                # In-flight requests across all DP replicas
MAX_MODEL_LEN=32768             # Server context; must cover prompt + generated tokens
# REGEN_LIMIT=1000              # Uncomment for a smoke test on the first N rows

# Steps 2-4 (prepare data, hidden-states server, training)
OUTPUT_DIR="$PERSIST/train/speculators/qwen3-8b-perfectblend-eagle3"
# Shards are a one-time artifact for parallel JSON loading (~7x faster):
# datasets parallelizes split generation across files, not within one file.
SHARDS_DIR="/dev/shm/regen_shards"
HS_PATH="/dev/shm/hs_qwen3_8b"  # transient hidden states, deleted after use
SEQ_LENGTH=8192                 # Data truncation length (prepare_data)
# Training pack length, decoupled from SEQ_LENGTH (RedHat recipe: prep 8-16k,
# pack 3-4k). 3072 + ttt16 + decay 0.867 aligns with DFlash (arXiv 2602.06036:
# seq 3072, block 16, loss decay gamma=7) and fits full-vocab logits on H200.
PACK_SEQ_LEN=3072
EPOCHS=3
LR=1e-4
VLLM_PORT=8000
# GPU split for online training: vLLM generates hidden states on-the-fly while
# the trainer consumes them, so the two need disjoint GPUs. 2 server replicas
# keep 6 training GPUs fed (5k example used a 1:1 ratio; 8B hidden-state gen is cheap).
VLLM_GPUS="0,1"
TRAIN_GPUS="2,3,4,5,6,7"
NUM_TRAIN_GPUS=6
# =======================================

# Step 1: Regenerate responses (on-policy data)
# Replaces the dataset's original responses with fresh generations from the
# target model, so the drafter learns what the target actually generates.
# --resume skips rows already present in the output JSONL; delete the file to
# start over. Sampling defaults (temperature 0.6, top_p 0.95, top_k 20) come
# from the model's generation_config.json via vLLM's --generation-config auto.
# Qwen3 is natively supported by vLLM/transformers: no --trust-remote-code.
# echo "=== Step 1: Regenerating responses ==="
# bash scripts/response_regeneration/run_all.sh \
#     --model "$MODEL" \
#     --dataset "$DATASET" \
#     --gpus "$GPUS" \
#     --tp-size "$TP_SIZE" \
#     --dp-size "$DP_SIZE" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --outfile "$REGEN_DATA" \
#     --concurrency "$CONCURRENCY" \
#     --resume \
#     ${REGEN_LIMIT:+--limit "$REGEN_LIMIT"}

# Step 2: Prepare data
# Regenerated rows are pre-tokenized (input_ids + loss_mask), so this step only
# truncates/filters, and writes token_freq.pt (used to build the draft vocab
# mapping automatically in Step 4).
# Idempotent: skip when a previous run already produced the artifacts —
# --overwrite would waste ~15 min re-shuffling 92GB of arrow data.
if [ -f "$OUTPUT_DIR/token_freq.pt" ]; then
    echo "=== Step 2: SKIPPED (found existing artifacts in $OUTPUT_DIR) ==="
else
    echo "=== Step 2: Preparing data ==="
    if [ ! -d "$SHARDS_DIR" ] || [ -z "$(ls -A "$SHARDS_DIR" 2>/dev/null)" ]; then
        echo "Sharding $REGEN_DATA into $SHARDS_DIR for parallel loading..."
        mkdir -p "$SHARDS_DIR"
        split -n l/16 --additional-suffix=.jsonl "$REGEN_DATA" "$SHARDS_DIR/part_"
    fi
    python scripts/prepare_data.py \
        --model "$MODEL" \
        --data "$SHARDS_DIR" \
        --output "$OUTPUT_DIR" \
        --seq-length "$SEQ_LENGTH" \
        --num-preprocessing-workers 16 \
        --overwrite
fi

# Step 3: Launch vLLM server in the background (generates hidden states on-the-fly)
echo "=== Step 3: Launching vLLM server for hidden states ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    -- --data-parallel-size 2 --port "$VLLM_PORT" --gpu-memory-utilization 0.85 &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 4: Train against the live vLLM server
# --on-missing generate --on-generate delete: hidden states are produced by the
# server per batch and discarded after use (no multi-TB cache on disk).
# Pack 3072 + ttt-steps 16 + decay 0.867 (= e^(-1/7)): aligned with DFlash's
# training setup (arXiv 2602.06036: seq 3072, block 16, loss decay gamma=7).
# wandb: needs WANDB_API_KEY in env (or `wandb login` once); WANDB_PROJECT /
# WANDB_ENTITY optional (default project "uncategorized").
RUN_NAME="qwen3-8b-perfectblend-eagle3"
[ -n "$COMMENT" ] && RUN_NAME="${RUN_NAME}_${COMMENT}"
RUN_NAME="${RUN_NAME}_{time}"
echo "=== Step 4: Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --hidden-states-path "$HS_PATH" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --muon-lr 2e-4 \
    --ttt-steps 16 \
    --ttt-step-loss-decay 0.867 \
    --total-seq-len "$PACK_SEQ_LEN" \
    --train-data-ratio 0.98 \
    --logger wandb \
    --run-name "$RUN_NAME" \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
