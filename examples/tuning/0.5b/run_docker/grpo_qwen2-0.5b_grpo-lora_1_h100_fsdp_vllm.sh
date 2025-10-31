#!/bin/bash
#SBATCH --gpus-per-task=a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=grpo_vrl
#SBATCH --output=job_output2.txt
#SBATCH --error=job_error2.txt
#SBATCH --ntasks=1
#SBATCH --mem=256Gb
#SBATCH --time=1:30:00

set -e
set -x  

source /home/mila/a/arnaud.bergeron1/scratch/verl/.env
podman pull docker-archive:${IMAGE_PATH}

podman run --rm --gpus=all \
  --mount type=bind,source=${VERL_PATH},destination=/home \
  docker.io/verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 \
  bash /home/examples/tuning/0.5b/run_docker/configs/grpo_run.sh
