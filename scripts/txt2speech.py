import argparse, os, sys, glob
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

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from text import text_to_sequence
from utils.tools import to_device

from g2p_en import G2p
from string import punctuation
import re
import yaml
from ldm.modules.fs2encoders.fastspeech2 import FastSpeech2
import soundfile
import librosa

sys.path.insert(0, '/work100/chenrenmiao/Project/230321-SpecVQGAN/hifigan-gradtts')
from meldataset import mel_spectrogram
from env import AttrDict
import json
from scipy.io.wavfile import write
from hifiganmodels import Generator as HiFiGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

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


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x
    
def get_phone_embedding_from_text(text, preprocess_config):
    ids = raw_texts = [text[:100]]
    speakers = np.array([0])
    texts = np.array([preprocess_english(text, preprocess_config)])
    text_lens = np.array([len(texts[0])])
    batch = (ids, raw_texts, speakers, texts, text_lens, max(text_lens))
    return batch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
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
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    # parser.add_argument(
    #     "--laion400m",
    #     action='store_true',
    #     help="uses the LAION400M model",
    # )
    # parser.add_argument(
    #     "--fixed_code",
    #     action='store_true',
    #     help="if enabled, uses the same starting code across samples ",
    # )
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
    # parser.add_argument(
    #     "--H",
    #     type=int,
    #     default=512,
    #     help="image height, in pixel space",
    # )
    # parser.add_argument(
    #     "--W",
    #     type=int,
    #     default=512,
    #     help="image width, in pixel space",
    # )
    # parser.add_argument(
    #     "--C",
    #     type=int,
    #     default=4,
    #     help="latent channels",
    # )
    # parser.add_argument(
    #     "--f",
    #     type=int,
    #     default=8,
    #     help="downsampling factor",
    # )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
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
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/tts-diffusion-inference.yaml",
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
        default="autocast"
    )
    opt = parser.parse_args()

    # if opt.laion400m:
    #     print("Falling back to LAION 400M model...")
    #     opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    #     opt.ckpt = "models/ldm/text2img-large/model.ckpt"
    #     opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    preprocess_config = yaml.load(
        open(config['fs2config']['preprocess_config'], "r"), Loader=yaml.FullLoader
    )
    # model_config = yaml.load(open(config['fs2config']['model_config'], "r"), Loader=yaml.FullLoader)

    # fs2model = FastSpeech2(preprocess_config, model_config).to(device)
    # fs2ckpt_path = config['fs2config']['ckpt_path']
    
    # fs2ckpt = torch.load(fs2ckpt_path)
    # fs2model.load_state_dict(fs2ckpt["model"])

    # fs2model.eval()
    # fs2model.requires_grad_ = False



    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "StableDiffusionV1"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    precision_scope = autocast if opt.precision=="autocast" else nullcontext


    HIFIGAN_CONFIG = '/work100/chenrenmiao/Project/230327-hifigan/UNIVERSAL_V1/config.json'
    HIFIGAN_CHECKPT = '/work100/chenrenmiao/Project/230327-hifigan/UNIVERSAL_V1/g_02500000'
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(get_phone_embedding_from_text('', preprocess_config))
                            uc = uc.repeat(opt.n_samples, 1, 1)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        c = model.get_learned_conditioning(get_phone_embedding_from_text(prompts[0], preprocess_config))

                        scale_num = 96
                        if c.shape[1] % scale_num != 0:
                            zeros = torch.zeros(1, scale_num - c.shape[1] % scale_num, c.shape[2],device=c.device)
                            c = torch.cat([c, zeros], dim=1)

                        c = c.repeat(opt.n_samples, 1, 1)
                        if opt.scale != 1.0:
                            uc = uc.repeat(1, c.shape[1] // uc.shape[1], 1)
                        print('c.shape', c.shape)

                        start_code = torch.randn([opt.n_samples, 4, c.shape[1] // 4, 20], device=device)
                        shape = [4, c.shape[1] // 4, 20]
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
