#!/usr/bin/env python
# coding: utf8

""" Unit testing for Separator class. """

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

from os.path import exists, join
from tempfile import TemporaryDirectory

from spleeter.audio.adapter import get_default_audio_adapter
from spleeter.separator import Separator

TEST_AUDIO_DESCRIPTOR = 'audio_example.mp3'
TEST_CONFIGURATIONS = {
    'spleeter:2stems': ('vocals', 'accompaniament'),
    'spleeter:4stems': ('vocals', 'drums', 'bass', 'other'),
    'spleeter:5stems': ('vocals', 'drums', 'bass', 'piano', 'other')
}


def test_separate():
    """ Test separation from raw data. """
    adapter = get_default_audio_adapter()
    waveform, _ = adapter.load(TEST_AUDIO_DESCRIPTOR)
    for configuration, instruments in TEST_CONFIGURATIONS.items():
        separator = Separator(configuration)
        prediction = separator.separate(waveform)
        assert len(prediction) == 2
        for instrument in instruments:
            assert instrument in prediction


def test_separate_to_file():
    """ Test file based separation. """
    for configuration, instruments in TEST_CONFIGURATIONS.items():
        separator = Separator(configuration)
        with TemporaryDirectory() as directory:
            separator.separate_to_file(
                TEST_AUDIO_DESCRIPTOR,
                directory)
            for instrument in instruments:
                assert exists(join(directory, '{}.wav'.format(instrument)))
                # TODO: Consider testing generated file as well.
