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

def plot_spectrogram(spec):
  plt.figure(figsize=(10, 4))
  plt.imshow(spec, aspect='auto', origin='lower')
  plt.colorbar()
  plt.title('Spectrogram')
  return None

def transform_spectrograms (spectrogram):
  spectrogram = torch.Tensor(spectrogram).T.unsqueeze(0)
  t1 = torchaudio.transforms.TimeStretch(n_freq=64, fixed_rate = 1.8)
  t2 = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)
  t3 = torchaudio.transforms.TimeMasking(time_mask_param=80)
  spectrogram = torch.abs(t1(spectrogram))
  spectrogram = t2(spectrogram)
  spectrogram = t3(spectrogram)
  #plot_spectrogram(spectrogram.squeeze(0))
  return spectrogram

def transform_captions(caption): 
  augmentor = RandAugment([
    en.add_synonyms,
    en.add_hypernyms,
    en.remove_articles,
    en.remove_contractions,
    en.remove_punctuation
    ], n=2, m=10, shuffle=False)
  
  for tx in augmentor:
    caption = tx(caption)
  #print("Augmented:", caption)
  return caption


def get_numpy_from_datadir(data_dir, split):
    
    """ Iterate through files in data_dir and load npy files into a list
        data_dir: directory containing the data
        split: train, val, or test
        tokenizer: tokenizer to use
        vocab_file: path to vocab file
        returns: tuple (encoded audio data , encoded text data)
    """
    # Initializing lists
    spectrograms = []
    captions = []
    train_spectrograms = []
    train_captions = []
    val_spectrograms = []
    val_captions = []
    test_spectrograms = []
    test_captions = []

    # Directories
    if split == 'train/val':
        data_dir =  data_dir + 'clotho_dataset_dev'
    elif split == 'test':
        data_dir =  data_dir + 'clotho_dataset_eva'

    # Load the data
    i = 0
    for file in tqdm(os.listdir(data_dir)):
    
        if file.endswith(".npy"):
              
            # Load the data item
            item = np.load(os.path.join(data_dir, file), allow_pickle=True)
            # Appending spectrogram
            spectrograms.append(item.features[0])
            # Cleaning caption
            caption = item.caption[0][6:-6] # Remove <s> and </s> tokens      
            captions.append(caption)
            i+=1

    if split == 'train/val':
        # Split spectrograms into train and val
        train_spectrograms = spectrograms[:int(len(spectrograms)*0.8)]
        val_spectrograms = spectrograms[int(len(spectrograms)*0.8):]
        # Split captions into train and val
        train_captions = captions[:int(len(captions)*0.8)]
        val_captions = captions[int(len(captions)*0.8):]
        
    if split == 'test':
        test_spectrograms = spectrograms
        test_captions = captions
    
    # Return dictionary of data with  keys
    return {'train_spectrograms': train_spectrograms, 'train_captions': train_captions, 'val_spectrograms': val_spectrograms, 'val_captions': val_captions, 'test_spectrograms': test_spectrograms, 'test_captions': test_captions}


# Dataloader that takes spectrogram and caption data to serve for training us
"""
    data_dir: Directory where the data is stored
    split: "dev" or "eva"
    tokenizer: Tokenizer to use for encoding the captions
    vocab_file: Path to the vocab file to use for encoding the captions (Optional)
"""

class AudioCaptioningDataset(Dataset):
    def __init__(self, spectrograms, captions, augment=False):

        self.spectogram_shape = (64, (40*44100+1)//512)     # From config file - 40 seconds * 44100 mhz
        self.caption_shape = (30,1)                         # From config file - 30 tokens long

        self.augment = augment
        self.spectrogram = spectrograms
        self.caption = captions

    def __len__(self):
        return len(self.spectrogram)

    def __getitem__(self, idx):
        spectrogram = self.spectrogram[idx]
        caption = self.caption[idx]

        if self.augment:
          # Augmentation
          spectrogram = transform_spectrograms(spectrogram)
          caption = transform_captions(caption)
        else:
          # Just transpose the spectrogram
          spectrogram = spectrogram.T
          spectrogram = torch.tensor(spectrogram).unsqueeze(0)

        # Padding the spectrogram
        spectrogram_pad = np.zeros(self.spectogram_shape)
        spectrogram_pad[:spectrogram.size()[1], :spectrogram.size()[2]] = spectrogram
        spectrogram = spectrogram_pad
        
        # Tokeninzing the captions
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer_output  = tokenizer(caption, return_tensors = 'pt', padding = 'max_length', max_length = self.caption_shape[0], truncation=True)
        caption = tokenizer_output['input_ids']
        attention_mask = tokenizer_output['attention_mask']
        
        return spectrogram, caption , attention_mask 


if __name__ == "__main__":

    data_dir = '/content/drive/My Drive/MusicCaptioning/dataset/'

    data_train = get_numpy_from_datadir(data_dir, 'train/val')
    data_test = get_numpy_from_datadir(data_dir, 'test')
    
    train_dataset = AudioCaptioningDataset(data_train['train_spectrograms'], data_train['train_captions'], augment = True)
    val_dataset = AudioCaptioningDataset(data_train['val_spectrograms'], data_train['val_captions'])
    test_dataset = AudioCaptioningDataset(data_test['test_spectrograms'], data_test['test_captions'])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    for (idx, batch) in enumerate(train_dataloader):
      pass
        
    for (idx, batch) in enumerate(val_dataloader):
      pass
    
    for (idx, batch) in enumerate(test_dataloader):
      pass


        

        
    
        

        
        
