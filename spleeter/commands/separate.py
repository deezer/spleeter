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

from ..audio.adapter import get_audio_adapter
from ..separator import Separator

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'



def entrypoint(arguments, params):
    """ Command entrypoint.

    :param arguments: Command line parsed argument as argparse.Namespace.
    :param params: Deserialized JSON configuration file provided in CLI args.
    """
    # TODO: check with output naming.
    audio_adapter = get_audio_adapter(arguments.audio_adapter)
    separator = Separator(
        arguments.configuration,
        MWF=arguments.MWF,
        stft_backend=arguments.stft_backend)
    for filename in arguments.inputs:
        separator.separate_to_file(
            filename,
            arguments.output_path,
            audio_adapter=audio_adapter,
            offset=arguments.offset,
            duration=arguments.duration,
            codec=arguments.codec,
            bitrate=arguments.bitrate,
            filename_format=arguments.filename_format,
            synchronous=False
        )
    separator.join()
