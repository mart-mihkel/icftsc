#!/usr/bin/env bash
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --cpus-per-task=32
#SBATCH --job-name="t5"
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --mem=32GB

BASE_MODELS=(
    google/flan-t5-small
    google/flan-t5-base
    google/flan-t5-large
    google/flan-t5-xl
    google/flan-t5-xxl
    google/t5gemma-2-270m-270m
    google/t5gemma-2-1b-1b
    google/t5gemma-2-4b-4b
)

PREFIX_INITS=(
    pretrained
    random
)

N_TRAIN_SAMPLES=20000
N_DEV_SAMPLES=1024
DATASET=multinerd
LOG_LEVEL=DEBUG
PREFIX_LR=1e-3
BATCH_SIZE=8
TASK=seq2seq
N_SHOT=3
EPOCHS=3

for BASE in ${BASE_MODELS[@]}; do
    uv run --no-sync cli few-shot \
        --batch-size $BATCH_SIZE \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --n-shot $N_SHOT \
        --model $BASE \
        --task $TASK

    uv run --no-sync cli fine-tune \
        --n-train-samples $N_TRAIN_SAMPLES \
        --n-dev-samples $N_DEV_SAMPLES \
        --batch-size $BATCH_SIZE \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --n-shot $N_SHOT \
        --no-head-only \
        --model $BASE \
        --task $TASK \
        --no-do-eval

    for PREFIX_INIT in ${PREFIX_INITS[@]}; do
        uv run --no-sync cli prompt-tune \
            --n-train-samples $N_TRAIN_SAMPLES \
            --n-dev-samples $N_DEV_SAMPLES \
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
