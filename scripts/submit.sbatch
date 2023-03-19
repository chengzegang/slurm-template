#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=168:00:00
#SBATCH --mem=300GB 
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=vit
#SBATCH --output=vit.out
#SBATCH --mail-type=END

module purge

singularity exec --nv \
	    --overlay /scratch/$USER/containers/overlay-50G-10M.ext3:ro \
		--overlay /scratch/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  		--overlay /scratch/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	    /bin/bash -c "source /ext3/env.sh; python -m project -m vit -t denoise -r /imagenet/train \
		--num-workers 16 --batch-size 128 --logdir logs/384 --max-epochs 600 \
		--ddp --warmup-epochs 30 --image-size 384 --patch-size 16 --split-ratio 0.25 0.05 0.7 \
		--refresh-rate 100 --zero --decoder-num-layers 12 --ddp"
