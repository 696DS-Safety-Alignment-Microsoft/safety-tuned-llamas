#!/bin/bash
#
#SBATCH --job-name=e_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH --mem=120000
#SBATCH --time=02-00:00:00

python safety-tuned-llamas/evals/compute_harmfulness.py \
    --input_json ./generation-artifacts/alpaca-small/BeaverTails-Generation \
    --output_dir ./evaluation-artifacts/alpaca-small/BeaverTails-Evaluation.csv
#python safety-tuned-llamas/examples/get_data.py