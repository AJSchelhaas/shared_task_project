#!/bin/bash
#SBATCH --job-name=BERT_Finetuning
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4000
#SBATCH --output=../logs/Peregrine/job-%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-user=jmvdheide96@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

module load Python
source VirtualEnv/bin/activate
pip install --upgrade pip
pip install transformers
pip install datasets

python run_mlm.py \
    --model_name_or_path ../model/toxic_classifier.model \
    --train_file /SemEval\ data/finetuning_train.txt \
    --validation_file /SemEval\ data/finetuning_trial.txt \
    --do_train \
    --do_eval \
    --output_dir /tmp/mlm_SemEval