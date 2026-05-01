#!/usr/bin/env bash

XS_DATA="aos"
F_PRECISION="single"

#SBATCH --job-name=pincell_benchmark
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x.out
#SBATCH --error=%x.err

module load nvidia/cuda/13.0.0

export OPENMC_CROSS_SECTIONS='some path'
