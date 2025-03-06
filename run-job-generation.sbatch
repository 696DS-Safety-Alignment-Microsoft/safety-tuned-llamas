#!/bin/bash
#
#SBATCH --job-name=g_job2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH --mem=120000
#SBATCH --time=02-00:00:00

python safety-tuned-llamas/generation/hf_generate_answers.py \
    --base_model '/datasets/ai/llama2/7B' \
    --input_path 'PKU-Alignment/BeaverTails-Evaluation' \
    --output_path './generation-artifacts' \
    --prompt_template_path safety-tuned-llamas/configs/alpaca.json \
    --lora_weights './training-artifacts/alpaca-safer-100' \
    --load_8bit
    
    #--input_path 'safety-tuned-llamas/data/evaluation/Q-Harm.json' \ (for generate_answers.py)
