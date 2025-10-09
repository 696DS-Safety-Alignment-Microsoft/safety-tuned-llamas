#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint vram40
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=cossim.txt

source venv/bin/activate

python category_sampling.py \
    --out "/cossim_200.json" \
    --num_samples 200 \
