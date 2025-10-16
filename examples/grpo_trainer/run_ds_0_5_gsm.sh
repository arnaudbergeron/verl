#!/bin/bash
#SBATCH --gpus-per-task=a100l:4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gsm_vrl
#SBATCH --output=job_output2.txt
#SBATCH --error=job_error2.txt
#SBATCH --ntasks=1
#SBATCH --mem=256Gb
#SBATCH --time=12:30:00

module load anaconda
conda activate verlhf
module load cuda/12.4.0


source /home/mila/a/arnaud.bergeron1/scratch/verl/.env
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

unset ROCR_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=XFORMERS

set -x
bash run_gemma.sh "trainer.n_gpus_per_node=1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 trainer.logger=['console'] critic.model.path=Qwen/Qwen2.5-0.5B-Instruct actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct data.train_batch_size=256 actor_rollout_ref.actor.ppo_mini_batch_size=64 actor_rollout_ref.actor.ppo_micro_batch_size=2 critic.ppo_micro_batch_size=2"
