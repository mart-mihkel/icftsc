#!/usr/bin/env bash
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --cpus-per-task=32
#SBATCH --job-name="bert"
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --mem=32GB

BASE_MODELS=(
    distilbert/distilbert-base-cased
    jhu-clsp/mmBERT-small
    jhu-clsp/mmBERT-base
    EuroBERT/EuroBERT-210m
    EuroBERT/EuroBERT-610m
    EuroBERT/EuroBERT-2.1B
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
TASK=seqcls
N_SHOT=3
EPOCHS=3

for BASE in ${BASE_MODELS[@]}; do
    uv run --no-sync cli fine-tune \
        --n-train-samples $N_TRAIN_SAMPLES \
        --n-dev-samples $N_DEV_SAMPLES \
        --batch-size $BATCH_SIZE \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --n-shot $N_SHOT \
        --model $BASE \
        --task $TASK \
        --no-do-eval \
        --head-only

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
