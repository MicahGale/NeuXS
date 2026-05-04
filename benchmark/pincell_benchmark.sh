#!/usr/bin/env bash

#SBATCH --job-name=pincell_benchmark
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=0
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=pincell_benchmark.%j.out
#SBATCH --error=pincell_benchmark.%j.err

module load nvidia/cuda/13.0.0
module load gcc/13.2.0



XS_DATA="aos"
F_PRECISION="single"
N_NEUTRONS=10000000
N_THREADS=256

./pincell "$XS_DATA" "$F_PRECISION" "$N_NEUTRONS" "$N_THREADS"