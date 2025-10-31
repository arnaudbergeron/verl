#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gsm_vrl
#SBATCH --output=job_output2.txt
#SBATCH --error=job_error2.txt
#SBATCH --ntasks=1
#SBATCH --mem=256Gb
#SBATCH --time=00:30:00

# module load anaconda
# conda activate verlhf
# module load cuda/12.4.0

podman  create --gpus all --net=host --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2
podman save -o verl_image.tar docker.io/verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2
# podman run -it --gpus=all --mount type=bind,source=/home/mila/a/arnaud.bergeron1/scratch/verl,destination=$SLURM_TMPDIR/home docker.io/verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2   bash