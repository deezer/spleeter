#!/usr/bin/env python
# coding: utf8

""" Unit testing for Separator class. """

__email__ = 'spleeter@deezer.com'
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

BACKENDS = ["tensorflow", "librosa"]
TEST_CONFIGURATIONS = {el:el for el in BACKENDS}

res_4stems = {
                "vocals": {
                    "SDR": 3.25e-05,
                    "SAR": -11.153575,
                    "SIR": -1.3849,
                    "ISR": 2.75e-05
                },
                "drums": {
                    "SDR": -0.079505,
                    "SAR": -15.7073575,
                    "SIR": -4.972755,
                    "ISR": 0.0013575
                },
                "bass":{
                    "SDR": 2.5e-06,
                    "SAR": -10.3520575,
                    "SIR": -4.272325,
                    "ISR": 2.5e-06
                },
                "other":{
                    "SDR": -1.359175,
                    "SAR": -14.7076775,
                    "SIR": -4.761505,
                    "ISR": -0.01528
                }
            }

def generate_fake_eval_dataset(path):
    """
        generate fake evaluation dataset
    """
    aa = get_default_audio_adapter()
    n_songs = 2
    fs = 44100
    duration = 3
    n_channels = 2
    rng = np.random.RandomState(seed=0)
    for song in range(n_songs):
        song_path = join(path, "test", f"song{song}")
        makedirs(song_path, exist_ok=True)
        for instr in ["mixture", "vocals", "bass", "drums", "other"]:
            filename = join(song_path, f"{instr}.wav")
            data = rng.rand(duration*fs, n_channels)-0.5
            aa.save(filename, data, fs)



@pytest.mark.parametrize('backend', TEST_CONFIGURATIONS)
def test_evaluate(backend):
    with TemporaryDirectory() as directory:
        generate_fake_eval_dataset(directory)
        p = create_argument_parser()
        arguments = p.parse_args(["evaluate", "-p", "spleeter:4stems", "--mus_dir", directory, "-B", backend])
        params = load_configuration(arguments.configuration)
        metrics = evaluate.entrypoint(arguments, params)
        for instrument, metric in metrics.items():
            for m, value in metric.items():
                assert np.allclose(np.median(value), res_4stems[instrument][m], atol=1e-3)