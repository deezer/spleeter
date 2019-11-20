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
import ffmpeg
import numpy as np
# pylint: enable=import-error

from .adapter import AudioAdapter
from .. import SpleeterError
from ..utils.logging import get_logger

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


def _to_ffmpeg_time(n):
    """ Format number of seconds to time expected by FFMPEG.
    :param n: Time in seconds to format.
    :returns: Formatted time in FFMPEG format.
    """
    m, s = divmod(n, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%09.6f' % (h, m, s)


class FFMPEGProcessAudioAdapter(AudioAdapter):
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
        if not isinstance(path, str):
            path = path.decode()
        try:
            probe = ffmpeg.probe(path)
        except ffmpeg._run.Error as e:
            raise SpleeterError(
                'An error occurs with ffprobe (see ffprobe output below)\n\n{}'
                .format(e.stderr.decode()))
        if 'streams' not in probe or len(probe['streams']) == 0:
            raise SpleeterError('No stream was found with ffprobe')
        metadata = next(
            stream
            for stream in probe['streams']
            if stream['codec_type'] == 'audio')
        n_channels = metadata['channels']
        if sample_rate is None:
            sample_rate = metadata['sample_rate']
        output_kwargs = {'format': 'f32le', 'ar': sample_rate}
        if duration is not None:
            output_kwargs['t'] = _to_ffmpeg_time(duration)
        if offset is not None:
            output_kwargs['ss'] = _to_ffmpeg_time(offset)
        process = (
            ffmpeg
            .input(path)
            .output('pipe:', **output_kwargs)
            .run_async(pipe_stdout=True, pipe_stderr=True))
        buffer, _ = process.communicate()
        waveform = np.frombuffer(buffer, dtype='<f4').reshape(-1, n_channels)
        if not waveform.dtype == np.dtype(dtype):
            waveform = waveform.astype(dtype)
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
        directory = os.path.split(path)[0]
        if not os.path.exists(directory):
            os.makedirs(directory)
        get_logger().debug('Writing file %s', path)
        input_kwargs = {'ar': sample_rate, 'ac': data.shape[1]}
        output_kwargs = {'ar': sample_rate, 'strict': '-2'}
        if bitrate:
            output_kwargs['audio_bitrate'] = bitrate
        if codec is not None and codec != 'wav':
            output_kwargs['codec'] = codec
        process = (
            ffmpeg
            .input('pipe:', format='f32le', **input_kwargs)
            .output(path, **output_kwargs)
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True))
        try:
            process.stdin.write(data.astype('<f4').tobytes())
            process.stdin.close()
            process.wait()
        except IOError:
            raise SpleeterError(f'FFMPEG error: {process.stderr.read()}')
        get_logger().info('File %s written', path)
