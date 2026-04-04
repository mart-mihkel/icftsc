#!/usr/bin/env bash
#SBATCH --nodelist=firefly1,firefly2
#SBATCH --output=out/slurm/%j-%x.out
#SBATCH --cpus-per-task=32
#SBATCH --job-name="run"
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --nodes=1

TASK=seqcls
BASE=distilbert/distilbert-base-cased
# BASE=jhu-clsp/mmBERT-small
# BASE=jhu-clsp/mmBERT-base
# BASE=EuroBERT/EuroBERT-210m
# BASE=EuroBERT/EuroBERT-610m
# BASE=EuroBERT/EuroBERT-2.1B

# TASK=causal
# BASE=openai-community/gpt2
# BASE=openai-community/gpt2-medium
# BASE=openai-community/gpt2-large
# BASE=openai-community/gpt2-xl
# BASE=openai-community/gpt2-xxl
# BASE=EleutherAI/pythia-70m
# BASE=EleutherAI/pythia-160m
# BASE=EleutherAI/pythia-410m
# BASE=EleutherAI/pythia-1b
# BASE=EleutherAI/pythia-1.4b
# BASE=EleutherAI/pythia-2.8b
# BASE=EleutherAI/pythia-6.9b
# BASE=google/gemma-3-270m-it
# BASE=google/gemma-3-1b-it
# BASE=google/gemma-3-4b-it
# BASE=Qwen/Qwen2.5-0.5B
# BASE=Qwen/Qwen2.5-1.5B
# BASE=Qwen/Qwen3.5-0.8B
# BASE=Qwen/Qwen3.5-2B
# BASE=Qwen/Qwen3.5-4B
# BASE=Qwen/Qwen3.5-9B
# BASE=meta-llama/Llama-3.2-1B
# BASE=meta-llama/Llama-3.2-3B
# BASE=meta-llama/Llama-3.2-1B-Instruct
# BASE=meta-llama/Llama-3.2-3B-Instruct
# BASE=meta-llama/Llama-3.1-8B
# BASE=meta-llama/Llama-3.1-8B-Instruct

# TASK=seq2seq
# BASE=google-t5/t5-small
# BASE=google-t5/t5-base
# BASE=google-t5/t5-large
# BASE=google-t5/t5-xl
# BASE=google-t5/t5-xxl
# BASE=google/flan-t5-small
# BASE=google/flan-t5-base
# BASE=google/flan-t5-large
# BASE=google/flan-t5-xl
# BASE=google/flan-t5-xxl
# BASE=google/t5gemma-2-270m-270m
# BASE=google/t5gemma-2-1b-1b
# BASE=google/t5gemma-2-4b-4b

NAME=$(echo $BASE | awk -F / '{print $2}')
N_TRAIN_SAMPLES=20000
N_DEV_SAMPLES=1024
DATASET=multinerd
LOG_LEVEL=DEBUG
PREFIX_LR=1e-3
BATCH_SIZE=8
N_SHOT=3
EPOCHS=5

uv run cli few-shot \
    --run-name $NAME-few-shot-$N_SHOT-shot-$DATASET \
    --experiment icftsc-$DATASET \
    --batch-size $BATCH_SIZE \
    --log-level $LOG_LEVEL \
    --dataset $DATASET \
    --n-shot $N_SHOT \
    --model $BASE \
    --task $TASK

# uv run cli fine-tune \
#     --run-name $NAME-cls-head-$N_SHOT-shot-$DATASET \
#     --n-train-samples $N_TRAIN_SAMPLES \
#     --n-dev-samples $N_DEV_SAMPLES \
#     --experiment icftsc-$DATASET \
#     --batch-size $BATCH_SIZE \
#     --log-level $LOG_LEVEL \
#     --dataset $DATASET \
#     --epochs $EPOCHS \
#     --n-shot $N_SHOT \
#     --model $BASE \
#     --task $TASK \
#     --no-do-eval \
#     --head-only

uv run cli fine-tune \
    --run-name $NAME-fine-tune-$N_SHOT-shot-$DATASET \
    --n-train-samples $N_TRAIN_SAMPLES \
    --n-dev-samples $N_DEV_SAMPLES \
    --experiment icftsc-$DATASET \
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
        --run-name $NAME-$PREFIX_INIT-prefix-$N_SHOT-shot-$DATASET \
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
