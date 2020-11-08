#!/usr/bin/env python
# coding: utf8

""" Unit testing for Separator class. """

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

import itertools
from os.path import splitext, basename, exists, join
from tempfile import TemporaryDirectory

import pytest
import numpy as np

import tensorflow as tf

from spleeter import SpleeterError
from spleeter.audio.adapter import get_default_audio_adapter
from spleeter.separator import Separator

TEST_AUDIO_DESCRIPTORS = ['audio_example.mp3', 'audio_example_mono.mp3']
BACKENDS = ["tensorflow", "librosa"]
MODELS = ['spleeter:2stems', 'spleeter:4stems', 'spleeter:5stems']

MODEL_TO_INST = {
    'spleeter:2stems': ('vocals', 'accompaniment'),
    'spleeter:4stems': ('vocals', 'drums', 'bass', 'other'),
    'spleeter:5stems': ('vocals', 'drums', 'bass', 'piano', 'other'),
}


MODELS_AND_TEST_FILES = list(itertools.product(TEST_AUDIO_DESCRIPTORS, MODELS))
TEST_CONFIGURATIONS = list(itertools.product(
    TEST_AUDIO_DESCRIPTORS, MODELS, BACKENDS))


print("RUNNING TESTS WITH TF VERSION {}".format(tf.__version__))


@pytest.mark.parametrize('test_file', TEST_AUDIO_DESCRIPTORS)
def test_separator_backends(test_file):
    adapter = get_default_audio_adapter()
    waveform, _ = adapter.load(test_file)

    separator_lib = Separator(
        "spleeter:2stems", stft_backend="librosa", multiprocess=False)
    separator_tf = Separator(
        "spleeter:2stems", stft_backend="tensorflow", multiprocess=False)

    # Test the stft and inverse stft provides exact reconstruction
    stft_matrix = separator_lib._stft(waveform)
    reconstructed = separator_lib._stft(
        stft_matrix, inverse=True, length=waveform.shape[0])
    assert np.allclose(reconstructed, waveform, atol=3e-2)

    # compare both separation, it should be close
    out_tf = separator_tf._separate_tensorflow(waveform, test_file)
    out_lib = separator_lib._separate_librosa(waveform, test_file)

    for instrument in out_lib.keys():
        # test that both outputs are close everywhere
        assert np.allclose(out_tf[instrument], out_lib[instrument], atol=1e-5)


@pytest.mark.parametrize('test_file, configuration, backend', TEST_CONFIGURATIONS)
def test_separate(test_file, configuration, backend):
    """ Test separation from raw data. """
    instruments = MODEL_TO_INST[configuration]
    adapter = get_default_audio_adapter()
    waveform, _ = adapter.load(test_file)
    separator = Separator(
        configuration, stft_backend=backend, multiprocess=False)
    prediction = separator.separate(waveform, test_file)
    assert len(prediction) == len(instruments)
    for instrument in instruments:
        assert instrument in prediction
    for instrument in instruments:
        track = prediction[instrument]
        assert waveform.shape[:-1] == track.shape[:-1]
        assert not np.allclose(waveform, track)
        for compared in instruments:
            if instrument != compared:
                assert not np.allclose(track, prediction[compared])


@pytest.mark.parametrize('test_file, configuration, backend', TEST_CONFIGURATIONS)
def test_separate_to_file(test_file, configuration, backend):
    """ Test file based separation. """
    instruments = MODEL_TO_INST[configuration]
    separator = Separator(
        configuration, stft_backend=backend, multiprocess=False)
    name = splitext(basename(test_file))[0]
    with TemporaryDirectory() as directory:
        separator.separate_to_file(
            test_file,
            directory)
        for instrument in instruments:
            assert exists(join(
                directory,
                '{}/{}.wav'.format(name, instrument)))


@pytest.mark.parametrize('test_file, configuration, backend', TEST_CONFIGURATIONS)
def test_filename_format(test_file, configuration, backend):
    """ Test custom filename format. """
    instruments = MODEL_TO_INST[configuration]
    separator = Separator(
        configuration, stft_backend=backend, multiprocess=False)
    name = splitext(basename(test_file))[0]
    with TemporaryDirectory() as directory:
        separator.separate_to_file(
            test_file,
            directory,
            filename_format='export/{filename}/{instrument}.{codec}')
        for instrument in instruments:
            assert exists(join(
                directory,
                'export/{}/{}.wav'.format(name, instrument)))


@pytest.mark.parametrize('test_file, configuration', MODELS_AND_TEST_FILES)
def test_filename_conflict(test_file, configuration):
    """ Test error handling with static pattern. """
    separator = Separator(configuration, multiprocess=False)
    with TemporaryDirectory() as directory:
        with pytest.raises(SpleeterError):
            separator.separate_to_file(
                test_file,
                directory,
                filename_format='I wanna be your lover')
