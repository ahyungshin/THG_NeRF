# generating head's mask.
python DataProcess/Gen_HeadMask.py --img_dir "train_data/cnn_imgs"

# generating 68-facial-landmarks by face-alignment, which is from 
# https://github.com/1adrianb/face-alignment
python DataProcess/Gen_Landmark.py --img_dir "train_data/cnn_imgs"

# generating the 3DMM parameters
python Fitting3DMM/FittingNL3DMM.py --img_size 512 \
                                    --intermediate_size 256  \
                                    --batch_size 9 \
                                    --img_dir "train_data/cnn_imgs"