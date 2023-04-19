import torchaudio
import numpy as np
audio_wavs, sr = torchaudio.load('/work100/chenrenmiao/Project/230302-vqvae/origin.wav')
audio_wavs = torchaudio.functional.resample(waveform=audio_wavs, orig_freq=sr, new_freq=16000).mean(0).unsqueeze(0)

from audio.tools import get_mel_from_wav
from audio.stft import TacotronSTFT
stft = TacotronSTFT(filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=64,
        sampling_rate=16000,
        mel_fmin=0,
        mel_fmax=8000,)
x, _, _ = get_mel_from_wav(audio_wavs.squeeze(), stft)

print(x.shape)
print(x)

