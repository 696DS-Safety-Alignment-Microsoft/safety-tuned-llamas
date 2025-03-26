#!/bin/bash
#
#SBATCH --job-name=t_job7
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH --mem=50GB
#SBATCH --time=02-00:00:00

python safety-tuned-llamas/training/finetuninig.py \
    --base_model '/datasets/ai/llama2/7B' \
    --data_path 'safety-tuned-llamas/data/training/subsample_50/alpaca_safer_50_dataset_7.json' \
    --output_dir './training-artifacts/alpaca-safer-50-dataset7' \
    --wandb_project 'safety-tuned-llms' \
    --batch_size 128 \
    --num_epochs 3
