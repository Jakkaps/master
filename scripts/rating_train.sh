#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=10:00:00
#SBATCH --job-name="dialog-dicriminator"
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --mem=32G
#SBATCH --constraint=gpu16g
#SBATCH --output=fine_tuning.txt

module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.8.0
source .venv/bin/activate
python rate_training.py --lr 0.001 --epochs 50 --batch_size 10 --n_layers=2 --graph_out_dim=10