#!/bin/bash

adv_estimation=(grpo)
outer_loop_size=(934 3736)
loss_name=(dpo_topr topr vanilla)
learning_rate=(1e-4 5e-5 1e-5)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for adv in "${adv_estimation[@]}"; do
  for outer_size in "${outer_loop_size[@]}"; do
    for loss in "${loss_name[@]}"; do
      for lr in "${learning_rate[@]}"; do
        sbatch --job-name="verl_${adv}_${outer_size}_${loss}" "${SCRIPT_DIR}/verl_run.sh" "$adv" "$outer_size" "$loss" "$lr"
      done
    done
  done
done