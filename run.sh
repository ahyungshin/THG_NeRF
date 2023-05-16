#!/bin/bash

#SBATCH --job-name=ll
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH -p batch_agi
#SBATCH --time=14-0
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G

. /data/shinahyung/anaconda3/etc/profile.d/conda.sh
conda activate adnerf

# training
python -u run.py --model_path "TrainedModels" \
                --model_file "model_Reso32HR.pth" \
                --data_path "/local_datasets/LipSync_datasets/preprocess_obama_512/dataset/Obama_512/0" \
                --save_root "./results" \

# rendering
# python -u run.py --model_path "results" \
#                 --model_file "Wav2NeRF_80000.pth" \
#                 --data_path "/local_datasets/LipSync_datasets/preprocess_obama_512/dataset/Obama_512/0/" \
#                 --save_root "./results_render" \
#                 --istest True