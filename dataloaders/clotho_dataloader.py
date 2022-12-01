import torch 
import torchaudio
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from transformers import RobertaTokenizer
from tqdm import tqdm
from niacin.augment import RandAugment
from niacin.text import en
import sys
import os
import matplotlib.pyplot as plt
import nltk
nltk.download('omw-1.4')


def transform_spectrograms (spectrogram):

  spectrogram_size = (64, (40*44100+1)//512)
  spectrogram = torch.Tensor(spectrogram).T.unsqueeze(0)

  t1 = torchaudio.transforms.TimeStretch(n_freq=64, fixed_rate = 1.8)
  t2 = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)
  t3 = torchaudio.transforms.TimeMasking(time_mask_param=80)

  spectrogram = torch.abs(t1(spectrogram))
  spectrogram = t2(spectrogram)
  spectrogram = t3(spectrogram)

  ret = np.zeros(spectrogram_size)
  ret[:spectrogram.size()[1], :spectrogram.size()[2]] = spectrogram

#   plt.figure(figsize=(10, 4))
#   plt.imshow(ret, aspect='auto', origin='lower')
#   plt.colorbar()
#   plt.title('Spectrogram')

  return ret

def transform_captions(caption):
  
  augmentor = RandAugment([
    en.add_synonyms,
    en.add_hypernyms,
    # en.add_hyponyms,
    # # en.add_misspelling,
    # en.add_contractions,
    # # en.add_fat_thumbs,
    en.remove_articles,
    # en.remove_characters,
    en.remove_contractions,
    en.remove_punctuation
    ], n=2, m=10, shuffle=False)


  
  for tx in augmentor:
    caption = tx(caption)

    
  return caption


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


    for file in tqdm(os.listdir(data_dir)):
    
        if file.endswith(".npy"):
              
            # Load the data item
            item = np.load(os.path.join(data_dir, file), allow_pickle=True)

            # Appending spectrogram
            spectrograms.append(item.features[0])

            # Cleaning caption
            caption = item.caption[0][6:-6] # Remove <s> and </s> tokens      
            captions.append(caption)
            

    return spectrograms, captions

# Dataloader that takes spectrogram and caption data to serve for training us
"""
    data_dir: Directory where the data is stored
    split: "dev" or "eva"
    tokenizer: Tokenizer to use for encoding the captions
    vocab_file: Path to the vocab file to use for encoding the captions (Optional)
"""


class AudioCaptioningDataset(Dataset):
    def __init__(self, data_dir, split, augment):

        if split == 'train/val':
            subfolder = 'clotho_dataset_dev'
        elif split == 'test':
            subfolder = 'clotho_dataset_eva'

        self.data_dir = data_dir + subfolder
        self.spectogram_shape = (64, (40*44100+1)//512 )     # From config file - 40 seconds * 44100 mhz
        self.caption_shape = (30,1)                         # From config file - 30 tokens long
        self.augment = augment
        self.split = split

        spectrogram, caption = get_data_from_numpy(self.data_dir, )
        self.spectrogram = spectrogram
        self.caption = caption

    def __len__(self):
        return len(self.spectrogram)

    def __getitem__(self, idx):
        
        spectrogram = self.spectrogram[idx]
        caption = self.caption[idx]

        if self.augment and self.split == 'train/val':
          spectrogram = transform_spectrograms(spectrogram)
          caption = transform_captions(caption)
        
        if not self.augment:
            spectrogram_pad = np.zeros(self.spectogram_shape)
            spectrogram_pad[:spectrogram.shape[1], :spectrogram.shape[0]] = spectrogram.T
            spectrogram = torch.tensor(spectrogram_pad)
        
        # Initialize the tokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer_output  = tokenizer(caption, return_tensors = 'pt', padding = 'max_length', max_length = 30, truncation=True)
        caption = tokenizer_output['input_ids']
        attention_mask = tokenizer_output['attention_mask']
        
        return spectrogram, caption , attention_mask 



if __name__ == "__main__":

    train_dataset = AudioCaptioningDataset(data_dir = '/home/nikhilrk/MusicCaptioning/MusicCaptioning/clotho-dataset/data/', 
                                       split='train/val',
                                       augment = True)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    data = next(iter(train_dataloader))
    print(data)

    
        




        

        
    
        

        
        
