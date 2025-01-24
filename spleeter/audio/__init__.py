#!/usr/bin/env python
# coding: utf8

"""
`spleeter.utils.audio` package provides various
tools for manipulating audio content such as :

- Audio adapter class for abstract interaction with audio file.
- FFMPEG implementation for audio adapter.
- Waveform convertion and transforming functions.
"""

# Python 3.11 is not backward compatible for String Enum
# https://tomwojcik.com/posts/2023-01-02/python-311-str-enum-breaking-change
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


class Codec(StrEnum):
    """Enumeration of supported audio codec."""

    WAV: str = "wav"
    MP3: str = "mp3"
    OGG: str = "ogg"
    M4A: str = "m4a"
    WMA: str = "wma"
    FLAC: str = "flac"
