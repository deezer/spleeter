#!/usr/bin/env python
# coding: utf8

""" Unit testing for Separator class. """

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

import filecmp

from os.path import splitext, basename, exists, join
from tempfile import TemporaryDirectory

import pytest
import numpy as np

from spleeter import SpleeterError
from spleeter.audio.adapter import get_default_audio_adapter
from spleeter.separator import Separator

TEST_AUDIO_DESCRIPTOR = 'audio_example.mp3'
TEST_AUDIO_BASENAME = splitext(basename(TEST_AUDIO_DESCRIPTOR))[0]
TEST_CONFIGURATIONS = [
    ('spleeter:2stems', ('vocals', 'accompaniment'), 'tensorflow'),
    ('spleeter:4stems', ('vocals', 'drums', 'bass', 'other'), 'tensorflow'),
    ('spleeter:5stems', ('vocals', 'drums', 'bass', 'piano', 'other'), 'tensorflow'),
    ('spleeter:2stems', ('vocals', 'accompaniment'), 'librosa'),
    ('spleeter:4stems', ('vocals', 'drums', 'bass', 'other'), 'librosa'),
    ('spleeter:5stems', ('vocals', 'drums', 'bass', 'piano', 'other'), 'librosa')
]


@pytest.mark.parametrize('configuration, instruments, backend', TEST_CONFIGURATIONS)
def test_separate(configuration, instruments, backend):
    """ Test separation from raw data. """
    adapter = get_default_audio_adapter()
    waveform, _ = adapter.load(TEST_AUDIO_DESCRIPTOR)
    separator = Separator(configuration, stft_backend=backend)
    prediction = separator.separate(waveform, TEST_AUDIO_DESCRIPTOR)
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
def test_separate_to_file(configuration, instruments, backend):
    """ Test file based separation. """
    separator = Separator(configuration, stft_backend=backend)
    with TemporaryDirectory() as directory:
        separator.separate_to_file(
            TEST_AUDIO_DESCRIPTOR,
            directory)
        for instrument in instruments:
            assert exists(join(
                directory,
                '{}/{}.wav'.format(TEST_AUDIO_BASENAME, instrument)))


@pytest.mark.parametrize('configuration, instruments, backend', TEST_CONFIGURATIONS)
def test_filename_format(configuration, instruments, backend):
    """ Test custom filename format. """
    separator = Separator(configuration, stft_backend=backend)
    with TemporaryDirectory() as directory:
        separator.separate_to_file(
            TEST_AUDIO_DESCRIPTOR,
            directory,
            filename_format='export/{filename}/{instrument}.{codec}')
        for instrument in instruments:
            assert exists(join(
                directory,
                'export/{}/{}.wav'.format(TEST_AUDIO_BASENAME, instrument)))


def test_filename_conflict():
    """ Test error handling with static pattern. """
    separator = Separator(TEST_CONFIGURATIONS[0][0])
    with TemporaryDirectory() as directory:
        with pytest.raises(SpleeterError):
            separator.separate_to_file(
                TEST_AUDIO_DESCRIPTOR,
                directory,
                filename_format='I wanna be your lover')
