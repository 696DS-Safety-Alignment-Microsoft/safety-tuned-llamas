#!/bin/bash
#
#SBATCH --job-name=e_job_9
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=[l40s]
#SBATCH --ntasks-per-node=1
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH --mem=50GB
#SBATCH --time=02-00:00:00

python safety-tuned-llamas/evals/compute_XS.py \
    --input_json ./generation-artifacts/llama2/alpaca-safer-150/alpaca-safer-150-dataset9/XSTest.json \
    --output_dir ./evaluation-artifacts/llama2/alpaca-safer-150/alpaca-safer-150-dataset9/XSTest.csv