#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 00:00:20
#SBATCH -J vlasiator_test
#SBATCH -o vlasiator_test.out.%j
#SBATCH -e vlasiator_test.out.%j
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH

./test.out
