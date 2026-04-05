#!/usr/bin/env bash
#SBATCH --nodelist=firefly1,firefly2
#SBATCH --output=out/slurm/%j-%x.out
#SBATCH --cpus-per-task=32
#SBATCH --job-name="gpt"
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --nodes=1

BASE_MODELS=(
    EleutherAI/pythia-70m
    EleutherAI/pythia-160m
    EleutherAI/pythia-410m
    EleutherAI/pythia-1b
    EleutherAI/pythia-1.4b
    EleutherAI/pythia-2.8b
    EleutherAI/pythia-6.9b
    google/gemma-3-270m-it
    google/gemma-3-1b-it
    google/gemma-3-4b-it
    Qwen/Qwen3-0.6B
    Qwen/Qwen3-1.7B
    Qwen/Qwen3-4B
    Qwen/Qwen3-8B
    meta-llama/Llama-3.2-1B
    meta-llama/Llama-3.2-3B
)

N_TRAIN_SAMPLES=20000
N_DEV_SAMPLES=1024
DATASET=multinerd
LOG_LEVEL=DEBUG
PREFIX_LR=1e-3
BATCH_SIZE=8
TASK=causal
N_SHOT=3
EPOCHS=3

for BASE in "${BASE_MODELS[@]}"; do
    uv run cli few-shot \
        --run-name $BASE/$N_SHOT-shot \
        --experiment icftsc-$DATASET \
        --batch-size $BATCH_SIZE \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --n-shot $N_SHOT \
        --model $BASE \
        --task $TASK

    uv run cli fine-tune \
        --n-train-samples $N_TRAIN_SAMPLES \
        --n-dev-samples $N_DEV_SAMPLES \
        --experiment icftsc-$DATASET \
        --run-name $BASE/fine-tune \
        --batch-size $BATCH_SIZE \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --n-shot $N_SHOT \
        --no-head-only \
        --model $BASE \
        --task $TASK \
        --no-do-eval

    for PREFIX_INIT in "pretrained" "random"; do
        uv run cli prompt-tune \
            --run-name $BASE/$PREFIX_INIT-prefix \
            --n-train-samples $N_TRAIN_SAMPLES \
            --n-dev-samples $N_DEV_SAMPLES \
            --experiment icftsc-$DATASET \
            --learning-rate $PREFIX_LR \
            --prefix-init $PREFIX_INIT \
            --batch-size $BATCH_SIZE \
            --log-level $LOG_LEVEL \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --n-shot $N_SHOT \
            --model $BASE \
            --task $TASK \
            --no-do-eval
    done
done
