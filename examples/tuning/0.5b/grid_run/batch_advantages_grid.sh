#!/bin/bash

adv_estimation=(question)
# Pairs are of form (batch_size outer_loop_size)
batch_outer_pairs=("32 7472" "8 934")
loss_name=(dpo_topr topr)
learning_rate=(1e-4 5e-5 1e-5)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for adv in "${adv_estimation[@]}"; do
  for pair in "${batch_outer_pairs[@]}"; do
    bsz=${pair%% *}
    outer_size=${pair#* }
    for loss in "${loss_name[@]}"; do
      for lr in "${learning_rate[@]}"; do
        sbatch --job-name="verl_${adv}_${outer_size}_${loss}_${bsz}" "${SCRIPT_DIR}/sbatch_scripts/verl_run.sh" "$adv" "$outer_size" "$loss" "$lr" "$bsz"
      done
    done
  done
done