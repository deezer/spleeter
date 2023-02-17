#!/usr/bin/env python
# coding: utf8

""" Unit testing for Separator class. """

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

from os import makedirs
from os.path import join
from tempfile import TemporaryDirectory

import numpy as np

from spleeter.__main__ import evaluate
from spleeter.audio.adapter import AudioAdapter

res_4stems = {
    "vocals": {"SDR": 3.25e-05, "SAR": -11.153575, "SIR": -1.3849, "ISR": 2.75e-05},
    "drums": {"SDR": -0.079505, "SAR": -15.7073575, "SIR": -4.972755, "ISR": 0.0013575},
    "bass": {"SDR": 2.5e-06, "SAR": -10.3520575, "SIR": -4.272325, "ISR": 2.5e-06},
    "other": {"SDR": -1.359175, "SAR": -14.7076775, "SIR": -4.761505, "ISR": -0.01528},
}


def generate_fake_eval_dataset(path):
    """
    Generate fake evaluation dataset
    """
    aa = AudioAdapter.default()
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
            data = rng.rand(duration * fs, n_channels) - 0.5
            aa.save(filename, data, fs)


def test_evaluate():
    with TemporaryDirectory() as dataset:
        with TemporaryDirectory() as evaluation:
            generate_fake_eval_dataset(dataset)
            metrics = evaluate(
                adapter="spleeter.audio.ffmpeg.FFMPEGProcessAudioAdapter",
                output_path=evaluation,
                params_filename="spleeter:4stems",
                mus_dir=dataset,
                mwf=False,
                verbose=False,
            )
            for instrument, metric in metrics.items():
                for m, value in metric.items():
                    assert np.allclose(
                        np.median(value), res_4stems[instrument][m], atol=1e-3
                    )
