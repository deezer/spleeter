#!/usr/bin/env python
# coding: utf8

""" Unit testing for Separator class. """

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

import filecmp
import itertools
import os
from os import makedirs
from os.path import splitext, basename, exists, join
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import json

import tensorflow as tf

from spleeter.audio.adapter import get_default_audio_adapter
from spleeter.commands import create_argument_parser

from spleeter.commands import train

from spleeter.utils.configuration import load_configuration

TRAIN_CONFIG = {
    "mix_name": "mix",
    "instrument_list": ["vocals", "other"],
    "sample_rate":44100,
    "frame_length":4096,
    "frame_step":1024,
    "T":128,
    "F":128,
    "n_channels":2,
    "chunk_duration":4,
    "n_chunks_per_song":1,
    "separation_exponent":2,
    "mask_extension":"zeros",
    "learning_rate": 1e-4,
    "batch_size":2,
    "train_max_steps": 10,
    "throttle_secs":20,
    "save_checkpoints_steps":100,
    "save_summary_steps":5,
    "random_seed":0,
    "model":{
        "type":"unet.unet",
        "params":{
               "conv_activation":"ELU",
               "deconv_activation":"ELU"
        }
    }
}


def generate_fake_training_dataset(path, instrument_list=["vocals", "other"]):
    """
        generates a fake training dataset in path:
        - generates audio files
        - generates a csv file describing the dataset
    """
    aa = get_default_audio_adapter()
    n_songs = 2
    fs = 44100
    duration = 6
    n_channels = 2
    rng = np.random.RandomState(seed=0)
    dataset_df = pd.DataFrame(columns=["mix_path"]+[f"{instr}_path" for instr in instrument_list]+["duration"])
    for song in range(n_songs):
        song_path = join(path, "train", f"song{song}")
        makedirs(song_path, exist_ok=True)
        dataset_df.loc[song, f"duration"] = duration
        for instr in instrument_list+["mix"]:
            filename = join(song_path, f"{instr}.wav")
            data = rng.rand(duration*fs, n_channels)-0.5
            aa.save(filename, data, fs)
            dataset_df.loc[song, f"{instr}_path"] = join("train", f"song{song}", f"{instr}.wav")

    dataset_df.to_csv(join(path, "train", "train.csv"), index=False)



def test_train():


    with TemporaryDirectory() as path:

        # generate training dataset
        generate_fake_training_dataset(path)

        # set training command aruments
        p = create_argument_parser()
        arguments = p.parse_args(["train", "-p", "useless_config.json", "-d", path])
        TRAIN_CONFIG["train_csv"] = join(path, "train", "train.csv")
        TRAIN_CONFIG["validation_csv"] = join(path, "train", "train.csv")
        TRAIN_CONFIG["model_dir"] = join(path, "model")
        TRAIN_CONFIG["training_cache"] = join(path, "cache", "training")
        TRAIN_CONFIG["validation_cache"] = join(path, "cache", "validation")

        # execute training
        res = train.entrypoint(arguments, TRAIN_CONFIG)

        # assert that model checkpoint was created.
        assert os.path.exists(join(path,'model','model.ckpt-10.index'))
        assert os.path.exists(join(path,'model','checkpoint'))
        assert os.path.exists(join(path,'model','model.ckpt-0.meta'))

if __name__=="__main__":
    test_train()