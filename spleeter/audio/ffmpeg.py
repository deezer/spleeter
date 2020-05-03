#!/usr/bin/env python
# coding: utf8

"""
    This module provides an AudioAdapter implementation based on FFMPEG
    process. Such implementation is POSIXish and depends on nothing except
    standard Python libraries. Thus this implementation is the default one
    used within this library.
"""

import os

# pylint: disable=import-error
import stempeg
import numpy as np
# pylint: enable=import-error

from .adapter import AudioAdapter
from .. import SpleeterError
from ..utils.logging import get_logger

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


class StempegProcessAudioAdapter(AudioAdapter):
    """ An AudioAdapter implementation that use FFMPEG binary through
    subprocess in order to perform I/O operation for audio processing.

    When created, FFMPEG binary path will be checked and expended,
    raising exception if not found. Such path could be infered using
    FFMPEG_PATH environment variable.
    """

    def load(
            self, path, offset=None, duration=None,
            sample_rate=None, dtype=np.float32):
        """ Loads the audio file denoted by the given path
        and returns it data as a waveform.

        :param path: Path of the audio file to load data from.
        :param offset: (Optional) Start offset to load from in seconds.
        :param duration: (Optional) Duration to load in seconds.
        :param sample_rate: (Optional) Sample rate to load audio with.
        :param dtype: (Optional) Numpy data type to use, default to float32.
        :returns: Loaded data a (waveform, sample_rate) tuple.
        :raise SpleeterError: If any error occurs while loading audio.
        """
        waveform, sample_rate = stempeg.read_streams(
            path=path,
            start=offset,
            duration=duration,
            stem_id=None,
            dtype=dtype,
            info=None,
            sample_rate=sample_rate,
            stems_from_multichannel=False
        )
        return (waveform, sample_rate)

    def save(
            self, path, data, sample_rate,
            codec=None, bitrate=None):
        """ Write waveform data to the file denoted by the given path
        using FFMPEG process.

        :param path: Path of the audio file to save data in.
        :param data: Waveform data to write.
        :param sample_rate: Sample rate to write file in.
        :param codec: (Optional) Writing codec to use.
        :param bitrate: (Optional) Bitrate of the written audio file.
        :raise IOError: If any error occurs while using FFMPEG to write data.
        """
        get_logger().debug('Writing file %s', path)
        stempeg.write_streams(
            path=path,
            data=data,
            sample_rate=sample_rate,
            codec=codec,
            bitrate=bitrate
        )
        get_logger().info('File %s written succesfully', path)
