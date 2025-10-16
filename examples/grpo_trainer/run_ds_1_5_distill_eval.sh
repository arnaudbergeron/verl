#!/bin/bash
#SBATCH --gpus-per-task=a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gsm_vrl
#SBATCH --output=job_output2.txt
#SBATCH --error=job_error2.txt
#SBATCH --ntasks=1
#SBATCH --mem=256Gb
#SBATCH --time=01:30:00

module load anaconda
conda activate verlhf
module load cuda/12.4.0


source /home/mila/a/arnaud.bergeron1/scratch/verl/.env
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

unset ROCR_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=XFORMERS

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /home/mila/a/arnaud.bergeron1/scratch/verl/checkpoints/verl_grpo_example_gsm8k/ds_1_5b_grpo_bf16_math_val_bs_16_full_4gpu_val_1epoch/global_step_40/actor \
    --target_dir /home/mila/a/arnaud.bergeron1/scratch/verl/checkpoints/verl_grpo_example_gsm8k/ds_1_5b_grpo_bf16_math_val_bs_16_full_4gpu_val_1epoch/global_step_40/saved_model


set -x
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/math/train.parquet \
    data.val_files=$HOME/data/math/test.parquet \
    data.max_prompt_length=766 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.path=/home/mila/a/arnaud.bergeron1/scratch/verl/checkpoints/verl_grpo_example_gsm8k/ds_1_5b_grpo_bf16_math_val_bs_16_full_4gpu_val_1epoch/global_step_40/saved_model \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    data.val_batch_size=1312 \
    data.train_batch_size=128 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=33534 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.lam=1.0 \
    algorithm.gamma=0.95 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.logger='["console","wandb"]' \
    trainer.experiment_name='ds_1_5b_grpo_bf16_math_val_bs_16_full_4gpu_val_1epoch_val' \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 $@
