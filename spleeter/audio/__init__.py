#!/usr/bin/env python
# coding: utf8

"""
    `spleeter.utils.audio` package provides various
    tools for manipulating audio content such as :

    - Audio adapter class for abstract interaction with audio file.
    - FFMPEG implementation for audio adapter.
    - Waveform convertion and transforming functions.
"""

from enum import Enum

# pyright: reportMissingImports=false
# pylint: disable=import-error
import tensorflow as tf
# pylint: enable=import-error

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


class Codec(str, Enum):
    """ Enumeration of supported audio codec. """

    WAV: str = 'wav'
    MP3: str = 'mp3'
    OGG: str = 'ogg'
    M4A: str = 'm4a'
    WMA: str = 'wma'
    FLAC: str = 'flac'


class STFTBackend(str, Enum):
    """ Enumeration of supported STFT backend. """

    AUTO: str = 'auto'
    TENSORFLOW: str = 'tensorflow'
    LIBROSA: str = 'librosa'

    @classmethod
    def resolve(cls: type, backend: str) -> str:
        backend = backend.upper()
        if backend not in cls.__members__.keys():
            raise ValueError(f'Unsupported backend {backend}')
        if backend == cls.AUTO:
            if len(tf.config.list_physical_devices('GPU')):
                return cls.TENSORFLOW
            return STFTBackend.LIBROSA
        return backend
