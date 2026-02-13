#!/bin/bash

pruned_model_dir=${1}
export CUDA_VISIBLE_DEVICES=${2:-"0,1,2,3"}
dataset_name=${3:-"m-a-p/CodeFeedback-Filtered-Instruction"}
router_finetune_steps=${4:-1000}
gradient_accumulation_steps=${5:-16}  # Increase to 128 to further reduce VRAM usage

echo "=================================================="
echo "Starting Router Fine-tuning"
echo "=================================================="
echo "Model directory: ${pruned_model_dir}"
echo "Dataset: ${dataset_name}"
echo "Fine-tuning steps: ${router_finetune_steps}"
echo "Gradient accumulation steps: ${gradient_accumulation_steps}"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Fine-tuning router on pruned model at ${pruned_model_dir}..."
python src/reap/finetune.py \
    --model-name ${pruned_model_dir} \
    --dataset-name ${dataset_name} \
    --router-finetune-steps ${router_finetune_steps} \
    --gradient-accumulation-steps ${gradient_accumulation_steps}

if [ $? -eq 0 ]; then
    echo "✅ Router fine-tuning completed successfully!"
else
    echo "❌ Router fine-tuning failed!"
    exit 1
fi
