import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
from tqdm import tqdm
import librosa
import numpy as np

import torchaudio as ta
import sys
sys.path.insert(0, '/work100/chenrenmiao/Project/230321-SpecVQGAN/hifigan-gradtts')
from meldataset import mel_spectrogram
from env import AttrDict
import json
from scipy.io.wavfile import write
from hifiganmodels import Generator as HiFiGAN

HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

def get_file_path(root_path, file_list, total_num):
    #获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        if len(file_list) >= total_num:
            return
        #获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        #判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            #递归获取所有文件和目录的路径
            get_file_path(dir_file_path, file_list, total_num)
        else:
            if dir_file_path[-3:] == "wav":
                file_list.append(dir_file_path)

# Add your custom dataset class here
class LibriTTSSpecs(Dataset):
    def __init__(self, 
                 split: str,
                 data_path: str, 
                 total_num: int,
                 duration: float,
                #  transform: Callable,
                **kwargs):
        # self.transforms = transform

        mel_path = '/work100/chenrenmiao/Project/230323-stablediffusion/mel_files/libriTTS.pt'
        if os.path.exists(mel_path):
            mels = torch.load(mel_path)
        else:
            file_list = []
            get_file_path(data_path, file_list, total_num)     
            mels = []
            for i in tqdm(file_list):
                audio_raw, sr = ta.load(i)
                audio = ta.transforms.Resample(orig_freq=sr, new_freq=22050)(audio_raw)
                target_length = int(duration * 22050)
                audio_length = audio.shape[-1]
                if audio_length < target_length:
                    pad_size = target_length - audio_length
                    audio = torch.nn.functional.pad(audio, (0, pad_size))
                elif audio_length > target_length:
                    audio = audio[:, :target_length]
                # Compute mel spectrogram
                # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
                S = mel_spectrogram(audio, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256,
                                            win_size=1024, fmin=0, fmax=8000, center=False).squeeze()
                # 增加一个维度
                # S = S[np.newaxis,...]
                
                mels.append(S)

            torch.save(mels,mel_path)

        self.mels = mels[:int(len(mels) * 0.8)] if split == "train" else mels[int(len(mels) * 0.8):]
    
    
    def __len__(self):
        return len(self.mels)
    
    def __getitem__(self, idx):
        mels = self.mels[idx]
        # max_val = np.max(mels)
        # min_val = np.min(mels)
        # mels = (mels - min_val) / (max_val - min_val)
        
        # if self.transforms is not None:
        #     mels = self.transforms(mels)
        return mels


class LibriTTSSpecsTrain(LibriTTSSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)

class LibriTTSSpecsValidation(LibriTTSSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('valid', **specs_dataset_cfg)

class LibriTTSSpecsTest(LibriTTSSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)


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
