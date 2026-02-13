#!/bin/bash

bash experiments/pruning-cli.sh

pruned_model_dir=$(grep "PRUNED_MODEL_DIR_PATH:" pruning_output.log | tail -1 | awk '{print $2}')

bash experiments/finetuning-cli.sh "${pruned_model_dir}"
