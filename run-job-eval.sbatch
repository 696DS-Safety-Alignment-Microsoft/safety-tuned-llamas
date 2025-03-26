#!/bin/bash
#
#SBATCH --job-name=e_job_7
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH --mem=120000
#SBATCH --time=02-00:00:00

python safety-tuned-llamas/evals/compute_harmfulness.py \
    --input_json ./generation-artifacts/alpaca-svd-100/alpaca-svd-100-dataset-low/BeaverTails-Evaluation \
    --output_dir ./evaluation-artifacts/alpaca-svd-100/alpaca-svd-100-dataset-low/BeaverTails-Evaluation.csv