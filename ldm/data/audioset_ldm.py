# AudioSet for VAE, this one for diffusion model

import numpy as np
from torch.utils.data import Dataset
import torchaudio as ta
import torch

import librosa
from utils.tools import pad_1D, pad_2D

import sys
sys.path.insert(0, '/data/home/scv9359/run/Code/hifi-gan')
from meldataset import mel_spectrogram

# Add your custom dataset class here
class AudioSetSpecs(Dataset):
    def __init__(
        self, filedir, filename, filetype
    ):
        self.filelist = []
        with open(f'{filedir}/{filename}') as lines:
            for i in lines:
                i = i.strip()
                self.filelist.append(f'{filedir}/{filetype}/{i}')

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        audio_data, sr = librosa.load(self.filelist[idx], sr=48000)
        limit_duration = 10

        duration = librosa.get_duration(audio_data, sr=sr)
        if duration > limit_duration:
            audio_data = audio_data[ : limit_duration * sr]

        audio_raw, sr = ta.load(self.filelist[idx])
        audio = ta.transforms.Resample(orig_freq=sr, new_freq=22050)(audio_raw)
        audio = audio.mean(dim=0, keepdim=True)
        if audio.shape[1] > limit_duration * 22050:
            audio = audio[:, :limit_duration * 22050]


        S = mel_spectrogram(audio, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256,
                              win_size=1024, fmin=0, fmax=8000, center=False).squeeze()

        y = S.shape[1]
        z = ((y - 1) // 4 + 1) * 4
        zeros = torch.zeros((S.shape[0], z - y))
        S = torch.cat((S, zeros), dim=1)

        S = torch.transpose(S, 0, 1)
        # S = S[np.newaxis,...]
        
    
        return {'mel': S, 'audio': audio_data}

    def reprocess(self, data, idxs):
        mels = [data[idx]["mel"] for idx in idxs]
        audios = [data[idx]["audio"] for idx in idxs]

        ori_mel_lens = max(np.array([mel.shape[0] for mel in mels]))
        scale_num = 4
        if ori_mel_lens % scale_num != 0:
            new_mel_lens = (ori_mel_lens // scale_num) * scale_num + scale_num
        else:
            new_mel_lens = ori_mel_lens
        
        audios = pad_1D(audios)
        mels = pad_2D(mels,maxlen=new_mel_lens)
        return {'mel': torch.from_numpy(mels), 'audio': torch.from_numpy(audios)}

    def collate_fn(self, data):
        data_size = len(data)
        idx_arr = np.arange(data_size)
        result = self.reprocess(data, idx_arr)

        
        # print('idx_arr', idx_arr)
        # print('len(idx_arr)', len(idx_arr))

        return result
    
class AudioSetSpecsTrain(AudioSetSpecs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class AudioSetSpecsValidation(AudioSetSpecs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class AudioSetSpecsTest(AudioSetSpecs):
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
