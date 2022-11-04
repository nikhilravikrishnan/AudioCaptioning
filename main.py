import yaml
import argparse
import os
from torch.utils.data.dataloader import DataLoader
from dataloaders.clotho_dataloader import AudioCaptioningDataset
from models.clip import BaseClip 
from transformers import ViTFeatureExtractor, ViTModel
import torch.optim 

parser = argparse.ArgumentParser(description="Music caption retrieval project for Georgia Tech CS7643")
parser.add_argument("--config", default="./configs/resnet.yaml")

def run_vision_transformer():
    """
    Make predictions on the dataset using a Vision Transformer model on Mel-Spectrogram image representations
    of input audio.

    See the original paper here:
    https://arxiv.org/pdf/2010.11929.pdf

    The implementation used comes from the HuggingFace transformers library and does not include a classification
    head by default. See their documentation here:
    https://huggingface.co/docs/transformers/model_doc/vit#vision-transformer-vit
    """
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
    epochs = args.epochs
    optimizer = torch.optim.adamw(lr=1e-3, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=1.0, factor=0.8)


    dataset = AudioCaptioningDataset(data_dir = args.data_dir, split=args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    model = BaseClip()
    
    

    # Training and Validation
    
    for e in range(epochs):
        
        # Training
        train_total_loss = 0
        model.train()
        for (idx,batch) in dataloader:
            
            batch_loss, _, __ = model.forward(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_total_loss += batch_loss.item()
        
        
        print('Training Loss:', train_total_loss/len(dataloader))
        print('Epoch:', e)


        model.eval()    
        val_total_loss = 0
        
        # Validation
        for (idx,batch) in dataloader:
            batch_loss, _, __ = model.forward(batch)
            val_total_loss += batch_loss.item()

        print('Validation Loss:', val_total_loss/len(dataloader))    

    
        

        



        

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
    if args.model == "ViT":
        run_vision_transformer()
    if args.model == "ResNet":
        run_resnet()
    if args.model == "Wavegram_Logmel_Cnn14":
        run_pann()
        

if __name__ == "__main__":
    main()