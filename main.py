import yaml
import argparse
import os
import torch
import models.clip

from transformers import ViTFeatureExtractor, ViTModel

parser = argparse.ArgumentParser(description="Music caption retrieval project for Georgia Tech CS7643")
parser.add_argument("--config", default="./configs/pann.yaml")

def run_vision_transformer(images):
    """
    Make predictions on the dataset using a Vision Transformer model on Mel-Spectrogram image representations
    of input audio.

    See the original paper here:
    https://arxiv.org/pdf/2010.11929.pdf

    The implementation used comes from the HuggingFace transformers library and does not include a classification
    head by default. See their documentation here:
    https://huggingface.co/docs/transformers/model_doc/vit#vision-transformer-vit
    """
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    clip_model = models.clip.CLIPModel(1, 768, 0, image_model, feature_extractor)
    clip_model(images)
    return

def run_resnet():
    """
    Make predictions on the dataset using a ResNet-50 model on Mel-Spectrogram image representations
    of input audio.

    The embedding for each image is made up of weights retrieved from the avgpool layer for that image after
    a forward pass through the network. This is done using the "create_feature_extractor" method from torchvision,
    but can also be done by setting a forward hook.

    (See this helpful post:
    https://stackoverflow.com/questions/52796121/how-to-get-the-output-from-a-specific-layer-from-a-pytorch-model)

    """
    return

def run_pann():
    """
    Make predictions on the dataset using a Pretrained Audio Neural Network
    (a CNN pretrained for audio classification using spectrogram images)

    See the original paper here:
    https://arxiv.org/pdf/1912.10211.pdf

    And GitHub repo here:
    https://github.com/qiuqiangkong/audioset_tagging_cnn
    """
    return

def main():
    # Use a config file to make sure we perform the correct experimental setup
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    # Get the dataset

    # Make predictions using the appropriate method for the selected model
    if args["model"] == "ViT":
        run_vision_transformer()
    if args["model"] == "ResNet":
        run_resnet()
    if args["model"] == "Wavegram_Logmel_Cnn14":
        run_pann()
        

if __name__ == "__main__":
    main()