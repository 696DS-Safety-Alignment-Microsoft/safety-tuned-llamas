#!/bin/bash
#
#SBATCH --job-name=g_job_0
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=[l40s]
#SBATCH --ntasks-per-node=1
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH --mem=50GB
#SBATCH --time=02-00:00:00

python safety-tuned-llamas/generation/generate_answers.py \
    --base_model '/datasets/ai/llama2/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9' \
    --input_path './safety-tuned-llamas/data/evaluation/I-Alpaca.json' \
    --output_path './generation-artifacts' \
    --prompt_template_path safety-tuned-llamas/configs/alpaca.json \
    --lora_weights './training-artifacts/llama2/alpaca-small' \
    --load_8bit
    
    #--input_path 'safety-tuned-llamas/data/evaluation/Q-Harm.json' \ (for generate_answers.py)
