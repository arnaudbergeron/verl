#!/bin/bash

adv_estimation=(grpo)
outer_loop_size=(1024 2048 4096)
loss_name=(dpo_topr topr grpo)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for adv in "${adv_estimation[@]}"; do
  for outer_size in "${outer_loop_size[@]}"; do
    for loss in "${loss_name[@]}"; do
      sbatch --job-name="verl_${adv}_${outer_size}_${loss}" ${SCRIPT_DIR}/verl_run.sh "$adv" "$outer_size" "$loss"
    done
  done
done