#!/bin/bash
#
#SBATCH --job-name=g_job_7
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=[l40s]
#SBATCH --ntasks-per-node=1
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH --mem=50GB
#SBATCH --time=02-00:00:00

python safety-tuned-llamas/generation/hf_generate_answers.py \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --input_path 'PKU-Alignment/BeaverTails-Evaluation' \
    --output_path './generation-artifacts' \
    --prompt_template_path safety-tuned-llamas/configs/alpaca.json \
    --lora_weights './training-artifacts/alpaca-svd-100-dataset-low' \
    --load_8bit
    
    #--input_path 'safety-tuned-llamas/data/evaluation/Q-Harm.json' \ (for generate_answers.py)
