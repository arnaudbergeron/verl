#!/bin/bash
#SBATCH --gpus-per-task=a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gsm_vrl
#SBATCH --output=job_output2.txt
#SBATCH --error=job_error2.txt
#SBATCH --ntasks=1
#SBATCH --mem=256Gb
#SBATCH --time=03:30:00

module load anaconda
conda activate verlhf
module load cuda/12.4.0


source /home/mila/a/arnaud.bergeron1/scratch/verl/.env

unset ROCR_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=XFORMERS

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/math/train.parquet \
    data.val_files=$HOME/data/math/test.parquet \
    data.max_response_length=8192 \
    data.val_batch_size=1312 \
    data.max_prompt_length=766 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=17408 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    trainer.logger=['wandb','console'] \
    trainer.val_only=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.82 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.max_num_batched_tokens=33534 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    critic.ppo_max_token_len_per_gpu=17408 \
    critic.model.use_remove_padding=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    critic.ppo_micro_batch_size=4 \
    critic.ulysses_sequence_parallel_size=1 \
    critic.model.fsdp_config.param_offload=False \
    +critic.model.fsdp_config.grad_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    trainer.critic_warmup=0 \
    +actor_rollout_ref.seed=1 \
    +critic.seed=1 \
    trainer.val_before_train=True \
    +trainer.rm_type=default \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=250 \
    trainer.test_freq=25 \
    trainer.project_name=verl_grpo_example_gsm8k \
    trainer.experiment_name='their_f32' \
    trainer.total_epochs=15