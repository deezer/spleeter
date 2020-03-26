#!/usr/bin/env python
# coding: utf8

""" Unit testing for Separator class. """

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

import filecmp
import itertools
from os.path import splitext, basename, exists, join
from tempfile import TemporaryDirectory

import pytest
import numpy as np

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


TEST_CONFIGURATIONS = list(itertools.product(TEST_AUDIO_DESCRIPTORS, MODELS, BACKENDS))


@pytest.mark.parametrize('configuration, instruments, backend', TEST_CONFIGURATIONS)
def test_separate(test_file, configuration, instruments, backend):
    """ Test separation from raw data. """
    adapter = get_default_audio_adapter()
    waveform, _ = adapter.load(test_file)
    separator = Separator(configuration, stft_backend=backend)
    prediction = separator.separate(waveform, test_file)
    assert len(prediction) == len(instruments)
    for instrument in instruments:
        assert instrument in prediction
    for instrument in instruments:
        track = prediction[instrument]
        assert waveform.shape == track.shape
        assert not np.allclose(waveform, track)
        for compared in instruments:
            if instrument != compared:
                assert not np.allclose(track, prediction[compared])


@pytest.mark.parametrize('configuration, instruments, backend', TEST_CONFIGURATIONS)
def test_separate_to_file(test_file, configuration, instruments, backend):
    """ Test file based separation. """
    separator = Separator(configuration, stft_backend=backend)
    basename = splitext(basename(test_file))
    with TemporaryDirectory() as directory:
        separator.separate_to_file(
            test_file,
            directory)
        for instrument in instruments:
            assert exists(join(
                directory,
                '{}/{}.wav'.format(basename, instrument)))


@pytest.mark.parametrize('configuration, instruments, backend', TEST_CONFIGURATIONS)
def test_filename_format(test_file, configuration, instruments, backend):
    """ Test custom filename format. """
    separator = Separator(configuration, stft_backend=backend)
    basename = splitext(basename(test_file))
    with TemporaryDirectory() as directory:
        separator.separate_to_file(
            test_file,
            directory,
            filename_format='export/{filename}/{instrument}.{codec}')
        for instrument in instruments:
            assert exists(join(
                directory,
                'export/{}/{}.wav'.format(basename, instrument)))


def test_filename_conflict(test_file):
    """ Test error handling with static pattern. """
    separator = Separator(TEST_CONFIGURATIONS[0][0])
    with TemporaryDirectory() as directory:
        with pytest.raises(SpleeterError):
            separator.separate_to_file(
                test_file,
                directory,
                filename_format='I wanna be your lover')
