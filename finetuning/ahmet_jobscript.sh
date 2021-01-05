#!/bin/bash
#SBATCH --job-name=BERT_Finetuning
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4000
#SBATCH --output=../logs/Peregrine/job-%j.log
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-user=jmvdheide96@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL


# Load Python module

module load Python/Python/3.7.4-GCCcore-8.3.0
module restore finetuning

module restore finetuning


# Set data location

DATADIR='/data/'"${USER}"'/python_envs'



# Prepare /data directories, ignore warning in case it already exists

mkdir -p "${DATADIR}"



# Create virtual environment

python3 -m venv "${DATADIR}"/python_envs



# Activate virtual environment

source "${DATADIR}"/python_envs/bin/activate



# Upgrade pip (inside virtual environment)

pip install --upgrade pip==20.0.2

pip install --upgrade setuptools==46.1.3



# Install required packages (inside virtual environment)

pip install transformers
pip install datasets

python run_mlm.py \
    --model_name_or_path ../model/toxic_classifier.model \
    --train_file ../Data/Source/tsd_train.csv \
    --validation_file ../Data/Source/tsd_trial.csv \
    --do_train \
    --do_eval \
    --output_dir tmp/mlm_SemEval
