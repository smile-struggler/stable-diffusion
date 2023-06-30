# AudioCaps for VAE, this one for diffusion model

import numpy as np
from torch.utils.data import Dataset
import torchaudio as ta
import torch
import pandas as pd
import os

import librosa
from utils.tools import pad_1D, pad_2D

import sys
sys.path.insert(0, '/data/home/scv9359/run/Code/hifi-gan')
from meldataset import mel_spectrogram
from transformers import GPT2Tokenizer, GPT2Model

# Add your custom dataset class here
class AudioCapsSpecs(Dataset):
    def __init__(
        self, filedir, metadir, filetype
    ):
        self.filelist = []
        self.captions = []
        df = pd.read_csv(f'{metadir}/{filetype}.csv')
        for index, row in df.iterrows():
            data_file = f'{filedir}/{filetype}/{row["youtube_id"]}.wav'
            if os.path.exists(data_file):
                self.filelist.append(data_file)
                self.captions.append(row["caption"])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        text = self.captions[idx]

        limit_duration = 10
        audio_raw, sr = ta.load(self.filelist[idx])
        audio = ta.transforms.Resample(orig_freq=sr, new_freq=22050)(audio_raw)
        audio = audio.mean(dim=0, keepdim=True)
        if audio.shape[1] > limit_duration * 22050:
            audio = audio[:, :limit_duration * 22050]


        S = mel_spectrogram(audio, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256,
                              win_size=1024, fmin=0, fmax=8000, center=False).squeeze()

        # y = S.shape[1]
        # z = ((y - 1) // 4 + 1) * 4
        # zeros = torch.zeros((S.shape[0], z - y))
        # S = torch.cat((S, zeros), dim=1)

        S = torch.transpose(S, 0, 1)
        # S = S[np.newaxis,...]
        
    
        return {'mel': S, 'text': text}

    def reprocess(self, data, idxs):
        mels = [data[idx]["mel"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]

        ori_mel_lens = min(np.array([mel.shape[0] for mel in mels]))
        scale_num = 16
        if ori_mel_lens % scale_num != 0:
            new_mel_lens = (ori_mel_lens // scale_num) * scale_num
        else:
            new_mel_lens = ori_mel_lens
        
        mels = np.stack([mel[:new_mel_lens, :] for mel in mels])
        return {'mel': torch.from_numpy(mels), 'txt': texts}

    def collate_fn(self, data):
        data_size = len(data)
        idx_arr = np.arange(data_size)
        result = self.reprocess(data, idx_arr)

        
        # print('idx_arr', idx_arr)
        # print('len(idx_arr)', len(idx_arr))

        return result
    
class AudioCapsSpecsTrain(AudioCapsSpecs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class AudioCapsSpecsValidation(AudioCapsSpecs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class AudioCapsSpecsTest(AudioCapsSpecs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# if __name__ == '__main__':
#     from omegaconf import OmegaConf

#     # SPECTROGRAMS + FEATURES
#     cfg = OmegaConf.load('./configs/vas_transformer.yaml')
#     data = instantiate_from_config(cfg.data)
#     data.prepare_data()
#     data.setup()
#     print(data.datasets['train'][24])
#     print(data.datasets['validation'][24])
#     print(data.datasets['validation'][-1]['feature'].shape)
#     print(data.datasets['validation'][-1]['image'].shape)
