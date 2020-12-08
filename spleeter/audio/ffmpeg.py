#!/usr/bin/env python
# coding: utf8

"""
    This module provides an AudioAdapter implementation based on FFMPEG
    process. Such implementation is POSIXish and depends on nothing except
    standard Python libraries. Thus this implementation is the default one
    used within this library.
"""

import os
import shutil

# pylint: disable=import-error
import stempeg
import numpy as np
# pylint: enable=import-error

from .adapter import AudioAdapter
from .. import SpleeterError
from ..utils.logging import get_logger

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


def _check_ffmpeg_install():
    """ Ensure FFMPEG binaries are available.

    :raise SpleeterError: If ffmpeg or ffprobe is not found.
    """
    for binary in ('ffmpeg', 'ffprobe'):
        if shutil.which(binary) is None:
            raise SpleeterError('{} binary not found'.format(binary))


def _to_ffmpeg_time(n):
    """ Format number of seconds to time expected by FFMPEG.
    :param n: Time in seconds to format.
    :returns: Formatted time in FFMPEG format.
    """
    m, s = divmod(n, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%09.6f' % (h, m, s)


def _to_ffmpeg_codec(codec):
    ffmpeg_codecs = {
        'm4a': 'aac',
        'ogg': 'libvorbis',
        'wma': 'wmav2',
    }
    return ffmpeg_codecs.get(codec) or codec


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
        waveform, sample_rate = stempeg.read_stems(
            path,
            start=offset,
            duration=duration,
            stem_id=None,
            dtype=dtype,
            info=None,
            sample_rate=sample_rate
        )
        return (waveform, sample_rate)

    def save(
            self, path, data, instruments, sample_rate,
            codec=None, bitrate=None):
        """ Write waveform data to the file denoted by the given path
        using FFMPEG process.

        :param path: Path of the audio file to save data in.
        :param data: Waveform data to write.
        :param instruments: Instrument labels.
        :param sample_rate: Sample rate to write file in.
        :param codec: (Optional) Writing codec to use.
        :param bitrate: (Optional) Bitrate of the written audio file.
        :raise IOError: If any error occurs while using FFMPEG to write data.
        """
        _check_ffmpeg_install()
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            raise SpleeterError(f'output directory does not exists: {directory}')
        get_logger().debug('Writing file %s', path)
        stempeg.write_stems(
            path,
            data=data,
            sample_rate=sample_rate,
            writer=stempeg.FilesWriter(
                codec=codec,
                bitrate=bitrate,
                multiprocess=True,
                stem_names=instruments
            )
        )
        get_logger().info('File %s written succesfully', path)
