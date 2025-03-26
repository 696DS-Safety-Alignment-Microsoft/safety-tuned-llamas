#!/bin/bash
#
#SBATCH --job-name=t_job_high
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=[l40s]
#SBATCH --ntasks-per-node=1
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH --mem=50GB
#SBATCH --time=02-00:00:00

python safety-tuned-llamas/training/finetuninig.py \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path 'safety-tuned-llamas/data/training/svd_100/alpaca_svd_100_dataset_high.json' \
    --output_dir './training-artifacts/alpaca-svd-100-dataset-high' \
    --wandb_project 'safety-tuned-llms' \
    --batch_size 128 \
    --num_epochs 3
