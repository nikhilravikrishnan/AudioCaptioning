import torch
import torch.nn as nn
from transformers import RobertaModel
# Using RoBERTa as the text encoding model
# https://huggingface.co/docs/transformers/model_doc/roberta

# http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.target_token_idx = 0 # Using the "<s>" token as the sentence embedding
        return

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids = input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        # Returns all elements of the requested token's embedding vector for each batch element
        return last_hidden_state[:, self.target_token_idx, :]