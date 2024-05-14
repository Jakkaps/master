#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=20:00:00
#SBATCH --job-name="dialog-dicriminator"
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --mem=100G
#SBATCH --constraint=gpu16g
#SBATCH --output=training.txt

module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.8.0
source .venv/bin/activate
python pre_training.py --lr 0.0001 --epochs 10 --batch_size 25 --n_layers=10 --graph_out_dim=10