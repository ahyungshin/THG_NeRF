#!/bin/bash

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
