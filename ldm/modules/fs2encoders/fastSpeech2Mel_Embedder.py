import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
from .fastspeech2 import FastSpeech2
import yaml

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class fastSpeech2Embedder(AbstractEncoder):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config, train_config, ckpt_path, device="cuda"):
        super(fastSpeech2Embedder, self).__init__()

        preprocess_config = yaml.load(
        open(preprocess_config, "r"), Loader=yaml.FullLoader
    )
        model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
        train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)

        self.model = FastSpeech2(preprocess_config, model_config).to(device)

        self.model_config = model_config
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt["model"])

        self.model.eval()
        self.model.requires_grad_ = False
        self.freeze()
        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        output = self.model.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.model.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.model.decoder(output, mel_masks)
        output = self.model.mel_linear(output)

        postnet_output = self.model.postnet(output) + output

        return postnet_output
    
    def encode(self, 
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,):
        
        import pdb
        # pdb.set_trace()

        return self(speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        p_targets,
        e_targets,
        d_targets,
        p_control,
        e_control,
        d_control,)


if __name__ == "__main__":
    from ldm.util import count_params
    model = fastSpeech2Embedder()
    count_params(model, verbose=True)