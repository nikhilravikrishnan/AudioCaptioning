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

"""
TODO:
1. Add the option to use a vocab file for encoding the captions
2. Preprocess the captions to add the <s> and </s> tokens
3. Resample spectrogram and add padding
"""
class AudioCaptioningDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer = 'roberta-base', vocab_file = None):
        
        SPLIT_PREFIX = f'clotho_dataset_{split}'
        self.data_dir = os.path.join(data_dir, SPLIT_PREFIX)

        # List of all the files in the dataset
        self.dataset = os.listdir(self.data_dir)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        

        # Load the file 
        data_item = np.load(os.path.join(self.data_dir, self.dataset[idx]), allow_pickle = True)
        caption = data_item.caption[0]
        spectrogram = data_item.features[0]
        
        # Clean the caption
        caption = caption[5:-5]
        
        # Encode the caption
        encoded_caption = RobertaTokenizer.from_pretrained('roberta-base').encode(caption)
        encoded_caption = np.array(encoded_caption)
        encoded_caption = torch.from_numpy(encoded_caption)

        # Convert the spectrogram to a tensor
        spectrogram = torch.from_numpy(spectrogram)

        return spectrogram, encoded_caption


class AudioCaptionDataloader(DataLoader):
    def __init__(self, data_dir, split, tokenizer = 'roberta-base', vocab_file = None):
        dataset = AudioCaptioningDataset(data_dir, split, tokenizer, vocab_file)
        super().__init__(dataset)






if __name__ == "__main__":

    test = AudioCaptioningDataset(data_dir = '/home/nikhilrk/MusicCaptioning/MusicCaptioning/clotho-dataset/data', split = 'dev', tokenizer = 'roberta-base')
    print(test.__getitem__(0))


        




        

        
    
        

        
        
