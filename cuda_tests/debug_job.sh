#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 00:00:20
#SBATCH -J cudatest
#SBATCH -o cudatest.out.%j
#SBATCH -e cudatest.out.%j
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH

cuda-memcheck test.out
