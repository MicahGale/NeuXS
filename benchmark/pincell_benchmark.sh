#!/usr/bin/env bash

XS_DATA="aos"
F_PRECISION="single"

#SBATCH --job-name=pincell_benchmark
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=%${XS_DATA}_${F_PRECISION}.out
#SBATCH --error=%${XS_DATA}_${F_PRECISION}.err

# if have run the setup.sh then the ENV is already ready to go

cd "${PWD}/../build/benchmark"
./pincell "$XS_DATA" "$F_PRECISION"