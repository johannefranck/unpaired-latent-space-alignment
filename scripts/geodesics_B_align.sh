#!/bin/bash
# scripts/geodesics_B_ld8_seed12_align.sh
#BSUB -J geo_B_ld8_s12_align
#BSUB -q gpua40
#BSUB -W 02:30
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o logs/geo_B_ld8_s12_align_%J.out
#BSUB -e logs/geo_B_ld8_s12_align_%J.err

module load cuda/11.7
source /dtu/blackhole/1d/155613/env/bin/activate
cd /dtu/blackhole/1d/155613/unpaired-latent-space-alignment

export HOME=$PWD
mkdir -p "$HOME/.cache" "$HOME/.config" logs

python build_mnist_geodesics.py \
  --device cuda \
  --experiment-name split42_digits012 \
  --model-name model_B \
  --model-seed 12 \
  --latent-dim 8 \
  --split-name align \
  --num-points 100 \
  --selection-mode random \
  --selection-seed 0 \
  --curve-type quadratic \
  --lr 0.05 \
  --steps 200 \
  --num-segments 20 \
  --print-every 0