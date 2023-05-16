# Talking Head Generation with NeRF

## Requirements
```
  conda env create -f environment.yaml
  conda activate wav2nerf
```
```
  git clone https://github.com/facebookresearch/pytorch3d.git
  cd pytorch3d && pip install -e . && cd ..
```

## Getting Started
Please download ConfigModels.zip, TrainedModels.zip, and LatentCodeSamples.zip from [HeadNeRF](https://github.com/CrisHY1995/headnerf). 

Please download wav2lip.pth from [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) and put in ./TrainedModels

## Usage
```
  bash run.sh
```

## Demo
To be updated

## Acknowledgments
This implementation is built on [HeadNeRF](https://github.com/CrisHY1995/headnerf). 
