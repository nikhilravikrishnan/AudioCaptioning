import sys
sys.path.append('/home/nikhilrk/MusicCaptioning/MusicCaptioning/')

import torch 
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import os
from transformers import RobertaTokenizer
import numpy as np
import models.text_encoder.TextEncoder as TextEncoder



# Dataloader that takes spectrogram and caption data to serve for training us
"""
data_dir: Directory where the data is stored
split: "dev" or "eva"
tokenizer: Tokenizer to use for encoding the captions
vocab_file: Path to the vocab file to use for encoding the captions (Optional)
"""

class AudioCaptioningDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer = 'roberta-base', vocab_file = None):
        
        SPLIT_PREFIX = f'clotho_dataset_{split}'
        self.data_dir = os.path.join(data_dir, SPLIT_PREFIX)

        # List of all the files in the dataset
        self.dataset = os.listdir(self.data_dir)

        self.spec_shape = ((30*44100+1)//512, 64)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        
        # Empty array of shape self.spec_shape
        spec = np.zeros(self.spec_shape)

        # Load the file 
        data_item = np.load(os.path.join(self.data_dir, self.dataset[idx]), allow_pickle = True)
        caption = data_item.caption[0]
        spectrogram = data_item.features[0]
        
        spec[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram
        
        # Clean the caption
        caption = caption[5:-5] 

        caption = TextEncoder.forward(caption)
        #Insert the <s> and </s> tokens
        # caption = f'<s> {caption} </s>'
        
        # Encode the caption
        # encoded_caption = RobertaTokenizer.from_pretrained('roberta-base').encode(caption)
        # encoded_caption = np.array(encoded_caption)
        # encoded_caption = torch.from_numpy(encoded_caption)



        # Convert the spectrogram to a tensor
        spec = torch.from_numpy(spec)
        caption = torch.from_numpy(np.array(caption))

        return spec, caption



# Collate function to pad the spectrograms to desired shape
def collate_fn(batch):
    raise NotImplementedError



if __name__ == "__main__":

    test = AudioCaptioningDataset(data_dir = '/home/nikhilrk/MusicCaptioning/MusicCaptioning/clotho-dataset/data', split = 'dev', tokenizer = 'roberta-base')
    dataloader = DataLoader(test, batch_size = 16, shuffle = True)
    spec, cap = next(iter(dataloader))
    print(spec.shape)
    print(cap.shape)
    print(len(dataloader))


        




        

        
    
        

        
        
