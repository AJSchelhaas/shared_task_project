#!/bin/bash
#SBATCH --job-name=SharedTask
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4000
#SBATCH --output=logs/Peregrine/job-%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-user=d.j.theodoridis@student.rug.nl
#SBATCH --mail-type=BEGIN,END,FAIL

module load Python/3.6.4-foss-2018a
source /data/s3121534/.envs/VirtualEnv/bin/activate
pip install --upgrade pip
pip install flair --user

export CUDA_VISIBLE_DEVICES=0

python classifier_BERT.py