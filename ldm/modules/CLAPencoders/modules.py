import numpy as np
import librosa
import torch
import laion_clap
import torch.nn as nn

# quantization
def int16_to_float32(x):
    return x.float() / 32767.0

def float32_to_int16(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).to(torch.int16)

# def int16_to_float32(x):
#     return (x / 32767.0).astype(np.float32)


# def float32_to_int16(x):
#     x = np.clip(x, a_min=-1., a_max=1.)
#     return (x * 32767.).astype(np.int16)

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenClapTextEmbedder(AbstractEncoder):
    """
    Uses the CLAP transformer encoder for text.
    """
    def __init__(self):
        super().__init__()
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        text_embed = self.model.get_text_embedding([text], use_tensor=True)
        text_embed = text_embed.squeeze()
        return text_embed 

    def encode(self, text):
        return self(text)


class FrozenClapAudioEmbedder(AbstractEncoder):
    """
        Uses the CLIP Audio encoder.
        """
    def __init__(self):
        super().__init__()
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()
        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, audio_data):
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
        audio_data = int16_to_float32(float32_to_int16(audio_data)) # quantize before send it in to the model
        audio_embed = self.model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
        audio_embed = audio_embed.unsqueeze(1)
        return audio_embed 

    def encode(self, audio_data):
        return self(audio_data)


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLAPTextEmbedder()
    count_params(model, verbose=True)