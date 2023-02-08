#!/bin/bash

#SBATCH -J run
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --time=4:00:00
#SBATCH --partition=t4v1,t4v2,rtx6000
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=logs/%x.%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90

/h/pwilson/anaconda3/envs/trusnet/bin/python main.py experiment=core_classification
