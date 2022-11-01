#!/bin/bash

# --- Download a pretrained model and save it to the checkpoint path ---
CHECKPOINT_PATH="../models/pretrained_weights/Wavegram_Logmel_Cnn14_mAP=0.439.pth"

wget -O $CHECKPOINT_PATH "https://zenodo.org/record/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth?download=1"