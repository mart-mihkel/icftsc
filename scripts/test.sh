#!/usr/bin/env bash
#SBATCH --output=out/slurm/%j-%x.out
#SBATCH --job-name="test"
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB

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
PREFIX_INIT=pretrained
DATASET=multinerd
LOG_LEVEL=DEBUG
PREFIX_LR=1e-3
BATCH_SIZE=8
N_SHOT=3
EPOCHS=1

if [[ $1 = few-shot ]]; then
    uv run cli few-shot \
        --run-name $NAME-few-shot-$N_SHOT-shot-$DATASET \
        --batch-size $BATCH_SIZE \
        --experiment icftsc-test \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --n-shot $N_SHOT \
        --model $BASE \
        --task $TASK

    exit 0
fi

if [[ $1 = cls-head ]]; then
    uv run cli fine-tune \
        --run-name $NAME-cls-head-$N_SHOT-shot-$DATASET \
        --batch-size $BATCH_SIZE \
        --experiment icftsc-test \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --n-shot $N_SHOT \
        --no-grad-chkpts \
        --model $BASE \
        --task $TASK \
        --head-only

    exit 0
fi

if [[ $1 = fine-tune ]]; then
    uv run cli fine-tune \
        --run-name $NAME-fine-tune-$N_SHOT-shot-$DATASET \
        --batch-size $BATCH_SIZE \
        --experiment icftsc-test \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --n-shot $N_SHOT \
        --no-grad-chkpts \
        --no-head-only \
        --model $BASE \
        --task $TASK

    exit 0
fi

if [[ $1 = prompt-tune ]]; then
    uv run cli prompt-tune \
        --run-name $NAME-$PREFIX_INIT-prefix-$N_SHOT-shot-$DATASET \
        --learning-rate $PREFIX_LR \
        --prefix-init $PREFIX_INIT \
        --experiment icftsc-test \
        --batch-size $BATCH_SIZE \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --n-shot $N_SHOT \
        --no-grad-chkpts \
        --model $BASE \
        --task $TASK

    exit 0
fi

echo "usage: $0 [few-shot|cls-head|fine-tune|prompt-tune]"
exit 1
