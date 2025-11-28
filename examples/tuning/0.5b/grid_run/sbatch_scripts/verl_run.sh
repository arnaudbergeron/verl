#!/bin/bash
#SBATCH --gpus-per-task=a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --mem=256Gb
#SBATCH --time=00:25:00

# Input arguments
adv_estimation=$1
outer_loop_size=$(($2))
loss_name=$3
learning_rate=$4
bsz=$(($5))
prob_granularity=$6
loss_agg=$7
test_freq=$((7472 / outer_loop_size))

# Load modules and activate conda environment
module load anaconda

set -a
source "${SCRATCH}/verl/.env"
set +a

env_name="${CONDA_ENV_NAME}"
conda activate "${env_name}"

module load cuda/12.4.0
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Run Logging Config
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NOW=$(date +%Y%m%d)
export WANDB_DIR=gsm8k-grpo-lora-qwen2.5-0.5b
export WANDB_PROJECT=${WANDB_DIR}
export WANDB_EXP=adv-${adv_estimation}-${loss_name}-o${outer_loop_size}-lr${learning_rate}-bsz${bsz}-prob-${prob_granularity}-${loss_agg}-${NOW}
MODEL_PATH=${SCRATCH}/verl/models/qwen_0.5B

# Main Training Loop
set -x
mini_batch_size=$(($bsz))
unset ROCR_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=${adv_estimation} \
        actor_rollout_ref.actor.policy_loss.loss_mode=${loss_name} \
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg} \
        actor_rollout_ref.actor.probability_granularity=${prob_granularity} \
        data.train_files=${SCRATCH}/verl/data/gsm8k/train.parquet \
        data.val_files=${SCRATCH}/verl/data/gsm8k/test.parquet \
        data.train_batch_size=${outer_loop_size} \
        data.val_batch_size=${mini_batch_size} \
        data.max_prompt_length=512 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=False \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        actor_rollout_ref.model.use_shm=True  \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.lora_rank=32 \
        actor_rollout_ref.model.lora_alpha=32 \
        actor_rollout_ref.model.target_modules=all-linear \
        actor_rollout_ref.actor.optim.lr=${learning_rate} \
        actor_rollout_ref.actor.optim.weight_decay=0.0 \
        actor_rollout_ref.actor.optim.clip_grad=1.0 \
        actor_rollout_ref.actor.use_torch_compile=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${mini_batch_size} \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.rollout.dtype=bfloat16 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=${mini_batch_size} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.rollout.val_kwargs.n=4 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.max_num_seqs=512 \
        actor_rollout_ref.rollout.max_model_len=1536 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.layered_summon=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size=${mini_batch_size} \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.entropy_coeff=0.00 \
        algorithm.kl_ctrl.kl_coef=0.0 \
        algorithm.use_kl_in_reward=False \
        trainer.val_before_train=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name=${WANDB_PROJECT} \
        trainer.experiment_name=${WANDB_EXP} \
        trainer.n_gpus_per_node=1 \
        trainer.rollout_data_dir=checkpoints/${WANDB_PROJECT}/${WANDB_EXP} \
        trainer.nnodes=1 \
        trainer.save_freq=2 \
        trainer.test_freq=${test_freq} \
        trainer.total_epochs=40
