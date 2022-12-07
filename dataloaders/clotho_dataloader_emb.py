import sys
sys.path.append("C:\\Users\\wabec\\MSA\\7643\\Project\\MusicCaptioning\\models")
import os
import torch 
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from transformers import RobertaTokenizer
import audio_encoders
import text_encoder

def get_data_from_numpy(data_dir, vocab_file = None, audio_encoder:str = None, layer_dict= None):
    
    """ Iterate through files in data_dir and load npy files into a list
        data_dir: directory containing the data
        split: train, val, or test
        tokenizer: tokenizer to use
        vocab_file: path to vocab file
        returns: tuple (encoded audio data , encoded text data)
    """
    # Load the data
    spectrograms = []
    captions = []
    spectrogram_length = (40*44100+1)//512 # FROM CONFIG FILE

    if audio_encoder == "ResNet50":
        cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif audio_encoder == "ViT_B_16":
        cnn = vit_b_16(ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    elif audio_encoder == "Cnn14":
        cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) #audio_encoder = audio_encoders.Cnn14()

    if vocab_file is not None:
        txt_encoder = text_encoder.TextEncoder()
        
    i = 0
    for file in os.listdir(data_dir):
        if i > 10:
            break

        if file.endswith(".npy"):

            # Load the data item
            item = np.load(os.path.join(data_dir, file), allow_pickle=True)
            # Padding and encoding spectrogram
            spectrogram = np.zeros((spectrogram_length, 64))
            spectrogram[:item.features[0].shape[0], :item.features[0].shape[1]] = item.features[0]
            
            if audio_encoder == "ResNet50":
                spectrogram = torch.from_numpy(spectrogram)
                spectrogram = torch.stack([spectrogram, spectrogram, spectrogram], dim=0)
                spectrogram = audio_encoders.get_audio_feature_vector(cnn, spectrogram, layer_dict)
            elif audio_encoder == "ViT_B_16":
                spectrogram = audio_encoders.get_vit_feature_vector(cnn, torch.from_numpy(spectrogram), layer_dict)
            elif audio_encoder == "Cnn14":
                spectrogram =  audio_encoders.get_audio_feature_vector(cnn, torch.from_numpy(spectrogram), layer_dict)

            spectrograms.append(spectrogram.detach().numpy())

            # Cleaning caption
            caption = item.caption[0][6:-6] # Remove <s> and </s> tokens
            # Initialize the tokenizer
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            tokenizer_output  = tokenizer(caption, return_tensors = 'pt', padding = 'max_length', max_length = 30)
            caption = txt_encoder(input_ids=tokenizer_output['input_ids'], attention_mask=tokenizer_output['attention_mask'])
            captions.append(caption.detach().numpy())
            i += 1
            print(i)
            
    # Concatenate all the data
    spectrograms = np.concatenate(spectrograms, axis=0)
    captions = np.concatenate(captions, axis=0)

    return torch.from_numpy(spectrograms), torch.from_numpy(captions)

# Dataloader that takes spectrogram and caption data to serve for training us
"""
    data_dir: Directory where the data is stored
    split: "dev" or "eva"
    tokenizer: Tokenizer to use for encoding the captions
    vocab_file: Path to the vocab file to use for encoding the captions (Optional)
"""


class AudioCaptioningDataset(Dataset):
    def __init__(self, data_dir, split, vocab_file:str = None, audio_encoder:str = None, layer_dict= None):

        if split == 'train/val':
            subfolder = 'clotho_dataset_dev'
        elif split == 'test':
            subfolder = 'clotho_dataset_eva'

        self.data_dir = data_dir + subfolder
        self.spectogram_shape = ((40*44100+1)//512, 64)     # From config file - 40 seconds * 44100 mhz
        self.caption_shape = (30,1)                         # From config file - 30 tokens long
        
        spectrogram, caption = get_data_from_numpy(self.data_dir, vocab_file = vocab_file, audio_encoder = audio_encoder, layer_dict = layer_dict)

        self.spectrogram = spectrogram
        self.caption = caption

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        spectrogram = self.spectrogram[idx]
        caption = self.caption[idx]
        return spectrogram, caption


        