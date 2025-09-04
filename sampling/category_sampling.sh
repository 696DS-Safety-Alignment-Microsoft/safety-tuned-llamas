#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint vram40
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=out.out
#SBATCH --error=out.err

source venv/bin/activate
python category_avg.py \
    --data_file 'category_wildguard.json'\
    --out "out/" \
    --num_samples 50 \
