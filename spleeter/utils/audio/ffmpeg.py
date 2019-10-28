#!/usr/bin/env python
# coding: utf8

"""
    This module provides an AudioAdapter implementation based on FFMPEG
    process. Such implementation is POSIXish and depends on nothing except
    standard Python libraries. Thus this implementation is the default one
    used within this library.
"""

import os
import os.path
import platform
import re
import subprocess

import numpy as np  # pylint: disable=import-error

from .adapter import AudioAdapter
from ..logging import get_logger

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

# Default FFMPEG binary name.
_UNIX_BINARY = 'ffmpeg'
_WINDOWS_BINARY = 'ffmpeg.exe'


def _which(program):
    """ A pure python implementation of `which`command
    for retrieving absolute path from command name or path.

    @see https://stackoverflow.com/a/377028/1211342

    :param program: Program name or path to expend.
    :returns: Absolute path of program if any, None otherwise.
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ['PATH'].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def _get_ffmpeg_path():
    """ Retrieves FFMPEG binary path using ENVVAR if defined
    or default binary name (Windows or UNIX style).

    :returns: Absolute path of FFMPEG binary.
    :raise IOError: If FFMPEG binary cannot be found.
    """
    ffmpeg_path = os.environ.get('FFMPEG_PATH', None)
    if ffmpeg_path is None:
        # Note: try to infer standard binary name regarding of platform.
        if platform.system() == 'Windows':
            ffmpeg_path = _WINDOWS_BINARY
        else:
            ffmpeg_path = _UNIX_BINARY
    expended = _which(ffmpeg_path)
    if expended is None:
        raise IOError(f'FFMPEG binary ({ffmpeg_path}) not found')
    return expended


def _to_ffmpeg_time(n):
    """ Format number of seconds to time expected by FFMPEG.

    :param n: Time in seconds to format.
    :returns: Formatted time in FFMPEG format.
    """
    m, s = divmod(n, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%09.6f' % (h, m, s)


def _parse_ffmpg_results(stderr):
    """ Extract number of channels and sample rate from
    the given FFMPEG STDERR output line.

    :param stderr: STDERR output line to parse.
    :returns: Parsed n_channels and sample_rate values.
    """
    # Setup default value.
    n_channels = 0
    sample_rate = 0
    # Find samplerate
    match = re.search(r'(\d+) hz', stderr)
    if match:
        sample_rate = int(match.group(1))
    # Channel count.
    match = re.search(r'hz, ([^,]+),', stderr)
    if match:
        mode = match.group(1)
        if mode == 'stereo':
            n_channels = 2
        else:
            match = re.match(r'(\d+) ', mode)
            n_channels = match and int(match.group(1)) or 1
    return n_channels, sample_rate


class _CommandBuilder(object):
    """ A simple builder pattern class for CLI string. """

    def __init__(self, binary):
        """ Default constructor. """
        self._command = [binary]

    def flag(self, flag):
        """ Add flag or unlabelled opt. """
        self._command.append(flag)
        return self

    def opt(self, short, value, formatter=str):
        """ Add option if value not None. """
        if value is not None:
            self._command.append(short)
            self._command.append(formatter(value))
        return self

    def command(self):
        """ Build string command. """
        return self._command


class FFMPEGProcessAudioAdapter(AudioAdapter):
    """ An AudioAdapter implementation that use FFMPEG binary through
    subprocess in order to perform I/O operation for audio processing.

    When created, FFMPEG binary path will be checked and expended,
    raising exception if not found. Such path could be infered using
    FFMPEG_PATH environment variable.
    """

    def __init__(self):
        """ Default constructor. """
        self._ffmpeg_path = _get_ffmpeg_path()

    def _get_command_builder(self):
        """ Creates and returns a command builder using FFMPEG path.

        :returns: Built command builder.
        """
        return _CommandBuilder(self._ffmpeg_path)

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
        """
        if not isinstance(path, str):
            path = path.decode()
        command = (
            self._get_command_builder()
            .opt('-ss', offset, formatter=_to_ffmpeg_time)
            .opt('-t', duration, formatter=_to_ffmpeg_time)
            .opt('-i', path)
            .opt('-ar', sample_rate)
            .opt('-f', 'f32le')
            .flag('-')
            .command())
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        buffer = process.stdout.read(-1)
        # Read STDERR until end of the process detected.
        while True:
            status = process.stderr.readline()
            if not status:
                raise OSError('Stream info not found')
            if isinstance(status, bytes):  # Note: Python 3 compatibility.
                status = status.decode('utf8', 'ignore')
            status = status.strip().lower()
            if 'no such file' in status:
                raise IOError(f'File {path} not found')
            elif 'invalid data found' in status:
                raise IOError(f'FFMPEG error : {status}')
            elif 'audio:' in status:
                n_channels, ffmpeg_sample_rate = _parse_ffmpg_results(status)
                if sample_rate is None:
                    sample_rate = ffmpeg_sample_rate
                break
        # Load waveform and clean process.
        waveform = np.frombuffer(buffer, dtype='<f4').reshape(-1, n_channels)
        if not waveform.dtype == np.dtype(dtype):
            waveform = waveform.astype(dtype)
        process.stdout.close()
        process.stderr.close()
        del process
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
        # NOTE: Tweak.
        if codec == 'wav':
            codec = None
        command = (
            self._get_command_builder()
            .flag('-y')
            .opt('-loglevel', 'error')
            .opt('-f', 'f32le')
            .opt('-ar', sample_rate)
            .opt('-ac', data.shape[1])
            .opt('-i', '-')
            .flag('-vn')
            .opt('-acodec', codec)
            .opt('-ar', sample_rate)  # Note: why twice ?
            .opt('-strict', '-2')     # Note: For 'aac' codec support.
            .opt('-ab', bitrate)
            .flag(path)
            .command())
        process = subprocess.Popen(
            command,
            stdout=open(os.devnull, 'wb'),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE)
        # Write data to STDIN.
        try:
            process.stdin.write(
                data.astype('<f4').tostring())
        except IOError:
            raise IOError(f'FFMPEG error: {process.stderr.read()}')
        # Clean process.
        process.stdin.close()
        if process.stderr is not None:
            process.stderr.close()
        process.wait()
        del process
        get_logger().info('File %s written', path)
