import sys

import os
import numpy as np

import argparse
import yaml
import torch

from data import Dataset as MyDataset

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer, seed_everything
from ldm.models.autoencoder import AutoencoderKL
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def get——