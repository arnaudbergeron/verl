#!/bin/bash

adv_estimation=(question)
# Pairs are of form (batch_size outer_loop_size)
batch_outer_pairs=("32 934")
loss_name=(dpo_topr)
learning_rate=(1e-5 1e-4 5e-5)
prob_granularity=(sequence cumultative_sequence token)
loss_agg=(token-mean seq-mean-token-sum-norm)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for adv in "${adv_estimation[@]}"; do
  for pair in "${batch_outer_pairs[@]}"; do
    bsz=${pair%% *}
    outer_size=${pair#* }
    for loss in "${loss_name[@]}"; do
      for lr in "${learning_rate[@]}"; do
        for granularity in "${prob_granularity[@]}"; do
          for l_agg in "${loss_agg[@]}"; do
            sbatch --job-name="verl_${adv}_${outer_size}_${loss}_${bsz}_${lr}_${granularity}_${l_agg}" "${SCRIPT_DIR}/sbatch_scripts/verl_run.sh" "$adv" "$outer_size" "$loss" "$lr" "$bsz" "$granularity" "$l_agg"
          done
        done
      done
    done
  done
done