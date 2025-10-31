set -ex
source /home/.env

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_DIR=${WANDB_DIR}
export WANDB_PROJECT=${WANDB_PROJECT}
export WANDB_EXP=${WANDB_EXP}
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=${WANDB_API_KEY}

SCRATCH_HOME="/home/mila/a/arnaud.bergeron1/scratch/verl"
MODEL_PATH="/home/models/qwen_0.5B"
NOW=$(date +%Y%m%d)
WANDB_DIR="gsm8k-grpo-lora-qwen2.5-0.5b"
WANDB_PROJECT=${WANDB_DIR}
WANDB_EXP="topr-4epoch-4096-bf16-zero-inside-docker-no-entropy-autograd-001lr-0.5b-${NOW}"

cd /home
pip install --no-deps -e .

nproc_per_gpu=32
nnodes=1
ngpu_per_node=1
total_procs=$(( nproc_per_gpu * nnodes * ngpu_per_node ))
mini_batch_size=$(( total_procs ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=no_estimator \
    actor_rollout_ref.actor.policy_loss.loss_mode=topr \
    data.train_files=/home/data/gsm8k/train.parquet \
    data.val_files=/home/data/gsm8k/test.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=${total_procs} \
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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${mini_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${mini_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXP} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=1 \
    trainer.total_epochs=4 $@ 
