#!/bin/bash
#
#SBATCH --job-name=t_job_5
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=[l40s]
#SBATCH --ntasks-per-node=1
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH --mem=50GB
#SBATCH --time=02-00:00:00

python safety-tuned-llamas/training/finetuninig.py \
    --base_model 'meta-llama/Llama-3.1-8B-Instruct' \
    --data_path 'safety-tuned-llamas/data/training/subsample_100/alpaca_safer_100_dataset_5.json' \
    --output_dir './training-artifacts/llama3/alpaca-safer-100/alpaca_safer_100_dataset_5' \
    --wandb_project 'safety-tuned-llms' \
    --batch_size 128 \
    --num_epochs 3
