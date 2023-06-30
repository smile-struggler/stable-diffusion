import argparse
import os
import sys
import glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchaudio as ta
import librosa
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from utils.tools import to_device

sys.path.insert(0, '/data/home/scv9359/run/Code/hifi-gan')
from meldataset import mel_spectrogram
from env import AttrDict
import json
from scipy.io.wavfile import write
from models import Generator as HiFiGAN

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio_path",
        type=str,
        nargs="?",
        default="audio.wav",
        help="path to the audio file",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/audio2img-samples",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action="store_true",
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action="store_true",
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given audio. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/audio2img-diffusion-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    opt = parser.parse_args()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    limit_duration = 10  # Set the duration limit

    audio_raw, sr = ta.load(opt.audio_path)
    if sr != 22050:
        audio = ta.transforms.Resample(orig_freq=sr, new_freq=22050)(audio_raw)
    else:
        audio = audio_raw
    audio = audio.mean(dim=0, keepdim=True)
    if audio.shape[1] > limit_duration * 22050:
        audio = audio[:, :limit_duration * 22050]

    data = [batch_size * [audio]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    HIFIGAN_CONFIG = "/data/home/scv9359/run/Code/hifi-gan/UNIVERSAL_V1/config.json"
    HIFIGAN_CHECKPT = "/data/home/scv9359/run/Code/hifi-gan/UNIVERSAL_V1/g_02500000"
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(
        torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)["generator"]
    )
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for audio_tmp in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            silence_audio = torch.zeros_like(audio_tmp[0])
                            uc = model.get_learned_conditioning(silence_audio)
                            uc = uc.repeat(opt.n_samples, 1, 1)
                        if isinstance(audio_tmp, tuple):
                            audio_tmp = list(audio_tmp)

                        c = model.get_learned_conditioning(audio_tmp[0])
                        print("???", audio_tmp[0])

                        c = c.repeat(opt.n_samples, 1, 1)
                        # if opt.scale != 1.0:
                        #     uc = uc.repeat(1, c.shape[1] // uc.shape[1], 1)
                        print('c.shape', c.shape)

                        start_code = torch.randn([opt.n_samples, 4, 864 // 4, 20], device=device)
                        shape = [4, 864 // 4, 20]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        print('samples_ddim.shape', samples_ddim.shape)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        print(x_samples_ddim.shape)
                        # x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        # print(x_samples_ddim.shape)
                        # x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        

                        # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        # x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                        # print(x_checked_image_torch.shape)

                        
                        for id,x_sample in enumerate(x_samples_ddim):
                            x_sample = np.squeeze(x_sample)
                            x_sample = x_sample.permute(1,0)
                            x_sample = x_sample.float()
                            # import pdb;pdb.set_trace()
                            audio_new = (vocoder.forward(x_sample.to(torch.device('cuda:0'))).float().cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                            write(f'./results/tts_scale3_{id}.wav', 22050, audio_new)
                            # y_inv = librosa.feature.inverse.mel_to_audio(x_sample.cpu().numpy(), sr=22050,n_fft=1024,
                            #     hop_length = 256, win_length = 1024)
                            # soundfile.write(f"./results/tts_Griffin-Lim_{id}.wav", y_inv, 22050)

                #         if not opt.skip_save:
                #             for x_sample in x_checked_image_torch:
                #                 x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                #                 img = Image.fromarray(x_sample.astype(np.uint8))
                #                 img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                #                 base_count += 1

                #         if not opt.skip_grid:
                #             all_samples.append(x_checked_image_torch)

                # if not opt.skip_grid:
                #     # additionally, save as grid
                #     grid = torch.stack(all_samples, 0)
                #     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                #     grid = make_grid(grid, nrow=n_rows)

                #     # to image
                #     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                #     img = Image.fromarray(grid.astype(np.uint8))
                #     img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                #     grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
