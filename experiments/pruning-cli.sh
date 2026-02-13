#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1:-"2,3"}
FIRST_DEVICE=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f1)
port=$((8000 + FIRST_DEVICE))
model_name=${2:-"unsloth/gpt-oss-20b-BF16"}
pruning_method=${3:-"reap"}
seed=${4:-42}
compression_ratio=${5:-0.5}
dataset_name=${6:-"m-a-p/CodeFeedback-Filtered-Instruction"}

# qa
run_lm_eval=${7:-true}
# coding
run_evalplus=${8:-true}
run_livecodebench=${9:-true}
# math
run_math=${10:-false}
# wildbench
run_wildbench=${11:-false}
singleton_super_experts=${12:-"false"}
singleton_outlier_experts=${13:-"false"}

# datasets
num_samples=${14:-1024}
split_by_category=${15:-"false"}
select_only_categories=${16:-"python"}
output_file_name="observations_${num_samples}_cosine-seed_${seed}.pt"


server_log_file_name="pruning-cli-${FIRST_DEVICE}.log"

echo "Running pruning with model: $model_name on devices: $CUDA_VISIBLE_DEVICES"
echo "Logs will be saved to: $server_log_file_name"
echo "Evaluations: lm_eval: $run_lm_eval, evalplus: $run_evalplus, livecodebench: $run_livecodebench math: $run_math, wildbench: $run_wildbench"
echo "Using seed: $seed"

echo "Running with model: $model_name, dataset: $dataset_name, compression ratio: $compression_ratio, pruning method: $pruning_method"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Step 1: Pruning the model from huggingface repository..."
# split-by-category is needed when using dataset: "m-a-p/CodeFeedback-Filtered-Instruction"
python src/reap/prune.py \
    --model-name $model_name \
    --dataset-name $dataset_name \
    --compression-ratio $compression_ratio \
    --prune-method $pruning_method \
    --profile false \
    --vllm_port $port \
    --server-log-file-name $server_log_file_name \
    --do-eval false \
    --distance_measure cosine \
    --seed $seed \
    --output_file_name ${output_file_name} \
    --singleton_super_experts ${singleton_super_experts} \
    --singleton_outlier_experts ${singleton_outlier_experts} \
    --samples_per_category ${num_samples} \
    --split-by-category ${split_by_category} \
    --record_pruning_metrics_only false \
    --finetune_router_after_prune false 2>&1 | tee pruning_output.log
# --select-only-categories ${select_only_categories} 

# Extract the actual pruned model directory from prune.py output
# This ensures path consistency between Python and Shell
echo "Extracting pruned model directory path..."
pruned_model_dir=$(grep "PRUNED_MODEL_DIR_PATH:" pruning_output.log | tail -1 | awk '{print $2}')

if [ -z "$pruned_model_dir" ]; then
    echo "Error: Could not extract pruned model directory path from prune.py output"
    exit 1
fi

echo "✅ Pruning completed successfully!"
echo "Pruned model saved to: ${pruned_model_dir}"
echo "OUTPUT_PATH:${pruned_model_dir}"

