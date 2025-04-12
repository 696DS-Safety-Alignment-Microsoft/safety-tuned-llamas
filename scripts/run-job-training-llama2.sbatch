#!/bin/bash
#
#SBATCH --job-name=t_job_low
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=[l40s]
#SBATCH --ntasks-per-node=1
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH --mem=50GB
#SBATCH --time=02-00:00:00

python safety-tuned-llamas/training/finetuninig.py \
    --base_model '/datasets/ai/llama2/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9' \
    --data_path 'safety-tuned-llamas/data/training/wildguard_annotations_50/instr_harm_sorted_T1_data_low.json' \
    --output_dir './training-artifacts/llama2/wildguard-annots/instr-harm-sorted-T1-low' \
    --wandb_project 'safety-tuned-llms' \
    --batch_size 128 \
    --num_epochs 3
