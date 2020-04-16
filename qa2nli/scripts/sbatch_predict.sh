#!/bin/bash
#SBATCH --job-name=dev-race-convert-train
#SBATCH -p 1080ti-long
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/race_convert_%j.out
#SBATCH --mem=100GB
cd /mnt/nfs/scratch1/dhruveshpate/nli_for_qa/qa-to-nli/qa2nli/scripts

export PYTHONPATH=$PYTHONPATH:/mnt/nfs/scratch1/dhruveshpate/nli_for_qa/qa-to-nli
python convert_race.py --device 0 --model_path "../../.models/bart_current/model.ckpt" --input_data ../../.data/RACE --set "dev" --output ../../.data/converted/RACE --postprocess_splitter period --overwrite
