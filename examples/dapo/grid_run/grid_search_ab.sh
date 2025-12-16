#!/bin/bash

adv_estimation=(question)
# Pairs are of form (batch_size outer_loop_size)
batch_outer_pairs=("32 32")
# batch_outer_pairs=("8 32")
#Care TIS GRPO
loss_name=(vanilla)
learning_rate=(1e-5 1e-6)
prob_granularity=(cumultative_sequence)
loss_agg=(seq-mean-token-sum-norm)
scale_ratio=(1)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
for adv in "${adv_estimation[@]}"; do
  for pair in "${batch_outer_pairs[@]}"; do
    bsz=${pair%% *}
    outer_size=${pair#* }
    for loss in "${loss_name[@]}"; do
      for lr in "${learning_rate[@]}"; do
        for granularity in "${prob_granularity[@]}"; do
          for l_agg in "${loss_agg[@]}"; do
            for sratio in "${scale_ratio[@]}"; do
              sbatch --job-name="verl_${adv}_${outer_size}_${loss}_${bsz}_${lr}_${granularity}_${l_agg}" "${SCRIPT_DIR}/sbatch_scripts/dapo_run.sh" "$adv" "$outer_size" "$loss" "$lr" "$bsz" "$granularity" "$l_agg" "$sratio"
              done
          done
        done
      done
    done
  done
done