import torch 
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import os
from transformers import RobertaTokenizer
import numpy as np



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

        self.spec_shape = ((40*44100+1)//512, 64)
        self.caption_shape = (30,1)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        
        
        # Empty array of shape self.spec_shape for zero padding
        spec = np.zeros(self.spec_shape)
        # caption = np.ones(self.caption_shape)*9 # 9 corresponds to <eos> token

        # Load the file 
        data_item = np.load(os.path.join(self.data_dir, self.dataset[idx]), allow_pickle = True)
        spectrogram = data_item.features[0]
        caption = data_item.caption[0]
        

        caption = caption[6:-6] # Remove the <s> and </s> tokens from the caption
        

        #Padding 
        spec[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram

        # Tokenize the caption
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenized_input = tokenizer(caption, return_tensors = 'pt', padding = 'max_length', max_length = 30, truncation=True)
        
        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']

        # Convert the spectrogram to a tensor
        spec = torch.from_numpy(spec)
        
        # Stack tensor to make it 3 channel
        spec = torch.stack([spec, spec, spec], dim = 0)

        return spec, input_ids, attention_mask 



# Collate function to pad the spectrograms to desired shape
def collate_fn(batch):
    raise NotImplementedError



if __name__ == "__main__":

    test = AudioCaptioningDataset(data_dir = '/home/nikhilrk/MusicCaptioning/MusicCaptioning/clotho-dataset/data', split = 'dev', tokenizer = 'roberta-base', vocab_file = '/home/nikhilrk/MusicCaptioning/MusicCaptioning/clotho-dataset/data/words_list.p')
    dataloader = DataLoader(test, batch_size = 16, shuffle = True)
    spec, ids, mask = next(iter(dataloader))
    print(spec.shape)
    print(ids.shape)
    print(mask.shape)
    print(len(dataloader))


        




        

        
    
        

        
        
