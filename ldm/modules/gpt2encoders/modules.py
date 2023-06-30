import numpy as np
import librosa
import torch
from transformers import GPT2Tokenizer, GPT2Model
import torch.nn as nn

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenGPT2Embedder(AbstractEncoder):
    """
    Uses the CLAP transformer encoder for text.
    """
    def __init__(self, device = 'cuda'):
        super().__init__()
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('/data/home/scv9359/run/Model/gpt2')
        self.model = GPT2Model.from_pretrained('/data/home/scv9359/run/Model/gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt', padding="max_length", truncation=True, max_length=77).to(self.device)
        if text == " ":
            encoded_input['input_ids'].fill_(50257)
            encoded_input['attention_mask'].fill_(0)
        output = self.model(**encoded_input)
        return output[0]

    def encode(self, text):
        return self(text)


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenGPT2Embedder()
    count_params(model, verbose=True)