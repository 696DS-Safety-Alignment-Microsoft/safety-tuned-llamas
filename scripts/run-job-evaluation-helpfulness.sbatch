#!/bin/bash
#
#SBATCH --job-name=e_job_3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH --mem=120000
#SBATCH --time=02-00:00:00

python safety-tuned-llamas/evals/compute_helpfulness.py \
    --input_json ./generation-artifacts/llama2/alpaca-small/I-Alpaca.json \
    --output_dir ./evaluation-artifacts/llama2/alpaca-small/I-Alpaca.csv