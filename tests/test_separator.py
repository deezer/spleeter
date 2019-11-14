#!/usr/bin/env python
# coding: utf8

""" Unit testing for Separator class. """

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

import filecmp

from os.path import exists, join
from tempfile import TemporaryDirectory

import pytest

from spleeter.audio.adapter import get_default_audio_adapter
from spleeter.separator import Separator

TEST_AUDIO_DESCRIPTOR = 'audio_example.mp3'
TEST_CONFIGURATIONS = [
    ('spleeter:2stems', ('vocals', 'accompaniment')),
    ('spleeter:4stems', ('vocals', 'drums', 'bass', 'other')),
    ('spleeter:5stems', ('vocals', 'drums', 'bass', 'piano', 'other'))
]


@pytest.mark.parametrize('configuration, instruments', TEST_CONFIGURATIONS)
def test_separate(configuration, instruments):
    """ Test separation from raw data. """
    adapter = get_default_audio_adapter()
    waveform, _ = adapter.load(TEST_AUDIO_DESCRIPTOR)
    separator = Separator(configuration)
    prediction = separator.separate(waveform)
    assert len(prediction) == len(instruments)
    for instrument in instruments:
        assert instrument in prediction
    for instrument in instruments:
        track = prediction[instrument]
        assert not (waveform == track).all()
        for compared in instruments:
            if instrument != compared:
                assert not (track == prediction[compared]).all()


@pytest.mark.parametrize('configuration, instruments', TEST_CONFIGURATIONS)
def test_separate_to_file(configuration, instruments):
    """ Test file based separation. """
    separator = Separator(configuration)
    with TemporaryDirectory() as directory:
        separator.separate_to_file(
            TEST_AUDIO_DESCRIPTOR,
            directory)
        for instrument in instruments:
            assert exists(join(directory, '{}.wav'.format(instrument)))
        for instrument in instruments:
            for compared in instrument:
                if instrument != compared:
                    assert not filecmp.cmp(
                        join(directory, '{}.wav'.format(instrument)),
                        join(directory, '{}.wav'.format(compared)))
