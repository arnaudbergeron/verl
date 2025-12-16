#!/bin/bash
#SBATCH --gpus-per-task=a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --mem=356Gb
#SBATCH --time=00:60:00

# Input arguments
adv_estimation=$1
outer_loop_size=$(($2))
loss_name=$3
learning_rate=$4
bsz=$(($5))
prob_granularity=$6
loss_agg=$7
sratio=$8
test_freq=$((3690 / outer_loop_size))

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
NOW=$(date +%Y%m%d)
export WANDB_DIR=dapo-lora-qwen2.5-1.5b
export WANDB_PROJECT=${WANDB_DIR}
export WANDB_EXP=adv-${adv_estimation}-${loss_name}-o${outer_loop_size}-lr${learning_rate}-bsz${bsz}-prob-${prob_granularity}-${loss_agg}-scale${sratio}-IS-correction--n16
MODEL_PATH=${SCRATCH}/verl/models/qwen_1.5B

# Main Training Loop
set -x
mini_batch_size=$(($bsz))
unset ROCR_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

dapo_train_path=${SCRATCH}/verl/data/dapo-17k/train.parquet
aime25_test_path=${SCRATCH}/verl/data/aime25/test.parquet
aime24_test_path=${SCRATCH}/verl/data/aime24/test.parquet
math500_test_path=${SCRATCH}/verl/data/math500/test.parquet
olympiad_test_path=${SCRATCH}/verl/data/olympiad/test.parquet
minerva_test_path=${SCRATCH}/verl/data/minerva/test.parquet

train_files="['$dapo_train_path']"
test_files="['$aime25_test_path', '$aime24_test_path', '$math500_test_path', '$olympiad_test_path', '$minerva_test_path']"

python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=${adv_estimation} \
        algorithm.norm_adv_by_std_in_grpo=False \
        actor_rollout_ref.actor.policy_loss.loss_mode=${loss_name} \
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg} \
        actor_rollout_ref.actor.probability_granularity=${prob_granularity} \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=${outer_loop_size} \
        data.val_batch_size=${mini_batch_size} \
        data.max_prompt_length=1500 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        actor_rollout_ref.model.use_shm=True  \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.lora_rank=32 \
        actor_rollout_ref.model.lora_alpha=32 \
        actor_rollout_ref.model.target_modules=all-linear \
        actor_rollout_ref.actor.optim.lr=${learning_rate} \
        actor_rollout_ref.actor.optim.weight_decay=0.0 \
        actor_rollout_ref.actor.optim.clip_grad=1.0 \
        actor_rollout_ref.actor.clip_ratio=${sratio} \
        actor_rollout_ref.actor.use_torch_compile=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${mini_batch_size} \
        actor_rollout_ref.actor.use_kl_loss=False \
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
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=500 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=4 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.max_num_seqs=1024 \
        actor_rollout_ref.rollout.max_model_len=1536 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.layered_summon=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size=${mini_batch_size} \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.entropy_coeff=0.00 \
        actor_rollout_ref.actor.tis_imp_ratio_cap=10 \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        algorithm.kl_ctrl.kl_coef=0.0 \
        algorithm.use_kl_in_reward=False \
        trainer.val_before_train=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name=${WANDB_PROJECT} \
        trainer.experiment_name=${WANDB_EXP} \
        trainer.n_gpus_per_node=${NUM_GPUS} \
        trainer.rollout_data_dir=checkpoints/${WANDB_PROJECT}/${WANDB_EXP} \
        trainer.nnodes=1 \
        trainer.save_freq=25 \
        trainer.test_freq=150 \
        trainer.total_epochs=1