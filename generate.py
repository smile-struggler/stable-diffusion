import yaml
import torch 
import librosa
from torchvision import transforms
import soundfile
import torchaudio as ta
import sys
from omegaconf import OmegaConf
import numpy as np
import torchvision
from copy import deepcopy

sys.path.insert(0, '/work100/chenrenmiao/Project/230321-SpecVQGAN/hifigan-gradtts')
from meldataset import mel_spectrogram
from env import AttrDict
import json
from scipy.io.wavfile import write
from hifiganmodels import Generator as HiFiGAN
from ldm.util import instantiate_from_config
from copy import deepcopy

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return {"model": model}, global_step

def get_model():
    # path_conf = '/work100/chenrenmiao/Project/230323-stablediffusion/logs/2023-03-28T13-32-38_melvae/configs/2023-03-28T13-32-38-project.yaml'
    # path_ckpt = '/work100/chenrenmiao/Project/230323-stablediffusion/logs/2023-03-28T13-32-38_melvae/checkpoints/epoch=000014.ckpt'

    path_conf = '/work100/chenrenmiao/Project/230323-stablediffusion/logs/2023-04-26T11-00-39_melvae/configs/2023-04-26T11-00-39-project.yaml'
    path_ckpt = '/work100/chenrenmiao/Project/230323-stablediffusion/logs/2023-04-26T11-00-39_melvae/checkpoints/epoch=000014.ckpt'
    
    config = OmegaConf.load(path_conf)
    model, step = load_model_from_config(config, path_ckpt)
    return model

HIFIGAN_CONFIG = '/work100/chenrenmiao/Project/230327-hifigan/UNIVERSAL_V1/config.json'
HIFIGAN_CHECKPT = '/work100/chenrenmiao/Project/230327-hifigan/UNIVERSAL_V1/g_02500000'

audio_raw, sr = ta.load("/work100/chenrenmiao/data/libritts/LibriTTS/train-clean-100/26/495/26_495_000005_000000.wav")
audio = ta.transforms.Resample(orig_freq=sr, new_freq=22050)(audio_raw)
target_length = int(7 * 22050)
audio_length = audio.shape[-1]
if audio_length < target_length:
    pad_size = target_length - audio_length
    audio = torch.nn.functional.pad(audio, (0, pad_size))
elif audio_length > target_length:
    audio = audio[:, : target_length]

# Compute mel spectrogram
# S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
S = mel_spectrogram(audio, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256,
                              win_size=1024, fmin=0, fmax=8000, center=False).squeeze()
    
test_S = S[np.newaxis,np.newaxis,...]
print(S.shape)
# test_S = S[np.newaxis,np.newaxis,...]
# print(test_S.shape)
# print(type(S))
# Reconstruct audio from mel spectrogram using Griffin-Lim
# y_inv = librosa.griffinlim(S)
# 使用Griffin-Lim算法将梅尔频谱转换回语音

# train_transforms = transforms.Compose([transforms.ToTensor()])
# img = train_transforms(S)
# 145
model = get_model()

model['model'].eval()

with torch.no_grad():
    posterior = model['model'].encode(test_S.to(torch.device('cuda:0')))
    z = posterior.sample()
    
    print('posterior.shape',posterior.mean.shape)
    # print('posterior.mean',posterior.mean)
    # print('posterior.std',posterior.std)
    # print('posterior.logvar',posterior.logvar)
    # print('z.content:',z)
    # z = z + torch.randn(z.shape).to(device=posterior.parameters.device)

    # new_posterior = deepcopy(posterior.mean)
    # # my_list = [i for i in range(20) if i not in [1, 3, 18]]
    # # print(my_list)
    # new_posterior[:,:,channel_num,:] =0
    # cha = new_posterior - posterior.mean
    # torch.set_printoptions(profile="full")
    # print("cha",cha)
    # print('new_posterior.shape',new_posterior.shape)
    # torch.set_printoptions(profile="default")
    reconstruction = model['model'].decode(z)

    # reconstruction = model['model'].forward(test_S.to(torch.device('cuda:0')))[0]
print('reconstruction.shape',reconstruction.shape)
reconstruction = np.squeeze(reconstruction)
# reconstruction = reconstruction.numpy()
print('reconstruction.shape',reconstruction.shape)
print(S)
print(reconstruction)

with open(HIFIGAN_CONFIG) as f:
    h = AttrDict(json.load(f))
vocoder = HiFiGAN(h)
vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
_ = vocoder.cuda().eval()
vocoder.remove_weight_norm()
with torch.no_grad():
    audio_new = (vocoder.forward(reconstruction.to(torch.device('cuda:0'))).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

print(audio_new)
write('./results/reconstruction.wav', 22050, audio_new)
# write('./results/audio_norm_mel.wav', 22050, audio_norm_mel)
print(type(audio))
ta.save('./results/origin.wav',audio,sample_rate=22050)
# y_inv = librosa.feature.inverse.mel_to_audio(S, sr=sr, n_fft=2048, hop_length=512, win_length=2048)
