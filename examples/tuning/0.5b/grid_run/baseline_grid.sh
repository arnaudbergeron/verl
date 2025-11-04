#!/bin/bash

adv_estimation=(baseline)
outer_loop_size=(934 1868 3736)
loss_name=(dpo_topr topr vanilla)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for adv in "${adv_estimation[@]}"; do
  for outer_size in "${outer_loop_size[@]}"; do
    for loss in "${loss_name[@]}"; do
      sbatch --begin=19:00 --job-name="verl_${adv}_${outer_size}_${loss}" "${SCRIPT_DIR}/verl_run.sh" "$adv" "$outer_size" "$loss"
    done
  done
done