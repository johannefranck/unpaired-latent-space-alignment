#!/bin/bash
#BSUB -J vae_B_ld8_test
#BSUB -q gpua40
#BSUB -W 01:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o logs/vae_B_ld8_test_%J.out
#BSUB -e logs/vae_B_ld8_test_%J.err

module load cuda/11.7

source /dtu/blackhole/1d/155613/env/bin/activate

cd /dtu/blackhole/1d/155613/unpaired-latent-space-alignment

export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config"

python train_vae.py \
  --device cuda \
  --data-root data \
  --split-file data/MNIST/splits/mnist_split_seed_42_digits_0_1_2.pt \
  --experiment-name split42_digits012 \
  --checkpoint-root checkpoints/mnist \
  --plot-root plots/mnist \
  --model-name model_B \
  --model-seed 12 \
  --epochs 100 \
  --batch-size 128 \
  --latent-dim 8 \
  --lr 1e-3