#!/usr/bin/env python
# coding: utf8

""" Unit testing for Separator class. """

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

import filecmp
import itertools
from os import makedirs
from os.path import splitext, basename, exists, join
from tempfile import TemporaryDirectory

import pytest
import numpy as np

import tensorflow as tf

from spleeter.audio.adapter import get_default_audio_adapter
from spleeter.commands import create_argument_parser

from spleeter.commands import evaluate

from spleeter.utils.configuration import load_configuration

res_4stems = {  "vocals": {
                    "SDR": -0.009,
                    "SAR": -19.044,
                    "SIR": -4.072,
                    "ISR": -0.000
                },
                "drums": {
                    "SDR": -0.066,
                    "SAR": -14.294,
                    "SIR": -4.908,
                    "ISR": 0.002
                },
                "bass":{
                    "SDR": -0.000,
                    "SAR": -6.364,
                    "SIR": -9.396,
                    "ISR": -0.001
                },
                "other":{
                    "SDR": -1.464,
                    "SAR": -14.893,
                    "SIR": -4.762,
                    "ISR": -0.027
                }
            }


def generate_fake_eval_dataset(path):
    aa = get_default_audio_adapter()
    n_songs = 2
    fs = 44100
    duration = 3
    n_channels = 2
    for song in range(n_songs):
        song_path = join(path, "test", f"song{song}")
        makedirs(song_path, exist_ok=True)
        rng = np.random.RandomState(seed=0)
        for instr in ["mixture", "vocals", "bass", "drums", "other"]:
            filename = join(song_path, f"{instr}.wav")
            data = rng.rand(duration*fs, n_channels)-0.5
            aa.save(filename, data, fs)


def test_evaluate(path="FAKE_MUSDB_DIR"):
    generate_fake_eval_dataset(path)
    p = create_argument_parser()
    arguments = p.parse_args(["evaluate", "-p", "spleeter:4stems", "--mus_dir", path])
    params = load_configuration(arguments.configuration)
    metrics = evaluate.entrypoint(arguments, params)
    for instrument, metric in metrics.items():
        for metric, value in metric.items():
            assert np.allclose(np.median(value), res_4stems[instrument][metric], atol=1e-3)
