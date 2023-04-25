# libritts for VAE, this one for diffusion model

import json
import math
import torch
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D
import yaml

# Add your custom dataset class here
class LibriTTSSpecs(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config
    ):
        preprocess_config = yaml.load(
            open(preprocess_config, "r"), Loader=yaml.FullLoader
        )
        train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        # self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)


        return {'image': torch.from_numpy(mel), 
                'txt': [
                    basename,
                    raw_text,
                    speaker,
                    phone,
                    phone.shape[0],
                    phone.shape[0],
                    mel,
                    mel.shape[0],
                    mel.shape[0],
                    pitch,
                    energy,
                    duration,
                ]}
    
    
    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    # def reprocess(self, data, idxs):
    #     ids = [data[idx]["id"] for idx in idxs]
    #     speakers = [data[idx]["speaker"] for idx in idxs]
    #     texts = [data[idx]["text"] for idx in idxs]
    #     raw_texts = [data[idx]["raw_text"] for idx in idxs]
    #     mels = [data[idx]["mel"] for idx in idxs]
    #     pitches = [data[idx]["pitch"] for idx in idxs]
    #     energies = [data[idx]["energy"] for idx in idxs]
    #     durations = [data[idx]["duration"] for idx in idxs]

    #     text_lens = np.array([text.shape[0] for text in texts])
    #     mel_lens = np.array([mel.shape[0] for mel in mels])

    #     speakers = np.array(speakers)
    #     texts = pad_1D(texts)
    #     mels = pad_2D(mels)
    #     pitches = pad_1D(pitches)
    #     energies = pad_1D(energies)
    #     durations = pad_1D(durations)

    #     return [
    #         ids,
    #         raw_texts,
    #         speakers,
    #         texts,
    #         text_lens,
    #         max(text_lens),
    #         mels,
    #         mel_lens,
    #         max(mel_lens),
    #         pitches,
    #         energies,
    #         durations,
    #     ]

    # def collate_fn(self, data):
    #     data_size = len(data)

    #     if self.sort:
    #         len_arr = np.array([d["text"].shape[0] for d in data])
    #         idx_arr = np.argsort(-len_arr)
    #     else:
    #         idx_arr = np.arange(data_size)

    #     # tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
    #     # idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
    #     # idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
    #     # if not self.drop_last and len(tail) > 0:
    #     #     idx_arr += [tail.tolist()]

    #     output = list()
    #     # for idx in idx_arr:
    #     result = self.reprocess(data, idx_arr)

        
    #     # print('idx_arr', idx_arr)
    #     # print('len(idx_arr)', len(idx_arr))

    #     return {'image': torch.from_numpy(result[6]), 'txt': result}
        

class LibriTTSSpecsTrain(LibriTTSSpecs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class LibriTTSSpecsValidation(LibriTTSSpecs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class LibriTTSSpecsTest(LibriTTSSpecs):
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
