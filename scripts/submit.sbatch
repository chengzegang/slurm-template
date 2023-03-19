#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=168:00:00
#SBATCH --mem=300GB 
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=vit
#SBATCH --output=vit.out
#SBATCH --mail-type=END

module purge

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun -w $head_node hostname -i)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export MASTER_ADDR=$head_node_ip

singularity exec --nv \
	    --overlay /scratch/$USER/containers/overlay-50G-10M.ext3:ro \
		--overlay /scratch/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  		--overlay /scratch/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; \
        torchrun \
        --nodes 2 \
        --nproc_per_node 1 \
        --rdzv_id $SLURM_JOB_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint $MASTER_ADDR:29500 \
        -m project train -m vit -t denoise -r /imagenet \
		--num-workers 16 --batch-size 128 --logdir logs/384 --max-epochs 600 \
		--ddp --warmup-epochs 30 --image-size 384 --patch-size 16 --split-ratio 0.25 0.05 0.7 \
		--refresh-rate 100 --zero --decoder-num-layers 12 --ddp"
