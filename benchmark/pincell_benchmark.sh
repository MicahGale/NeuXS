#!/usr/bin/env bash

#SBATCH --job-name=pincell_benchmark_array
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=0
#SBATCH --cpus-per-task=1
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=pincell_benchmark.%j.out
#SBATCH --error=pincell_benchmark.%j.err

module load nvidia/cuda/13.0.0
module load gcc/13.2.0

cd ~/NeuXS

if [ ! -d "build" ]; then
	bash setup.sh
fi

XS_DATA="aos"
F_PRECISION="single"
N_NEUTRONS=10000000
N_THREADS=256

cd build/benchmark

declare -a xs_types=("aos" "soa" "log" "slbw")

declare -a nums_parts=(100 500 1000 5000 10000 50000 100000 500000 1000000 5000000)

for XS_DATA in "${xs_types[@]}"
do
	for N_NEUTRONS in "${nums_parts[@]}"
	do
		./pincell "$XS_DATA" "$F_PRECISION" "$N_NEUTRONS" "$N_THREADS"
	done
done
