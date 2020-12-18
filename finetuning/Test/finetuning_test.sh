#!/bin/bash
#SBATCH --job-name=BERT_Finetuning
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4000
#SBATCH --output=logs/job-%j.log
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-user=jmvdheide96@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

#source VirtualEnv/bin/activate
#pip install --upgrade pip
#pip install transformers
#pip install datasets

python run_mlm.py \
    --model_name_or_path ../../model/toxic_classifier.model \
    --train_file finetuning_train.csv \
    --validation_file finetuning_trial.csv \
    --do_train \
    --do_eval \
    --output_dir tmp/mlm_SemEval
