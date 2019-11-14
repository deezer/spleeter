#!/usr/bin/env python
# coding: utf8

"""
    Entrypoint provider for performing source separation.

    USAGE: python -m spleeter separate \
        -p /path/to/params \
        -i inputfile1 inputfile2 ... inputfilen
        -o /path/to/output/dir \
        -i /path/to/audio1.wav /path/to/audio2.mp3
"""

from multiprocessing import Pool
from os.path import isabs, join, split, splitext
from tempfile import gettempdir

# pylint: disable=import-error
import tensorflow as tf
import numpy as np
# pylint: enable=import-error

from ..audio.adapter import get_audio_adapter
from ..audio.convertor import to_n_channels
from ..separator import Separator
from ..utils.estimator import create_estimator
from ..utils.tensor import set_tensor_shape

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


def entrypoint(arguments, params):
    """ Command entrypoint.

    :param arguments: Command line parsed argument as argparse.Namespace.
    :param params: Deserialized JSON configuration file provided in CLI args.
    """
    # TODO: check with output naming.
    audio_adapter = get_audio_adapter(arguments.audio_adapter)
    separator = Separator(arguments.configuration, arguments.MWF)
    for filename in arguments.audio_filenames:
        separator.separate_to_file(
            filename,
            arguments.output_path,
            audio_adapter=audio_adapter,
            offset=arguments.offset,
            duration=arguments.max_duration,
            codec=arguments.codec,
            bitrate=arguments.bitrate,
            synchronous=False
        )
    separator.join()
