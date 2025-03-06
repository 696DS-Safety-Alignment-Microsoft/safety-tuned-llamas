#!/bin/bash
#
#SBATCH --job-name=i_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH --mem=80000
#SBATCH --time=02-00:00:00

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"