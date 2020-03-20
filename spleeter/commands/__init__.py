#!/usr/bin/env python
# coding: utf8

""" This modules provides spleeter command as well as CLI parsing methods. """

import json
import logging
from argparse import ArgumentParser
from tempfile import gettempdir
from os.path import exists, join

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'



# -i opt specification (separate).
OPT_INPUT = {
    'dest': 'inputs',
    'nargs': '+',
    'help': 'List of input audio filenames',
    'required': True
}

# -o opt specification (evaluate and separate).
OPT_OUTPUT = {
    'dest': 'output_path',
    'default': join(gettempdir(), 'separated_audio'),
    'help': 'Path of the output directory to write audio files in'
}

# -f opt specification (separate).
OPT_FORMAT = {
    'dest': 'filename_format',
    'default': '{filename}/{instrument}.{codec}',
    'help': (
        'Template string that will be formatted to generated'
        'output filename. Such template should be Python formattable'
        'string, and could use {filename}, {instrument}, and {codec}'
        'variables.'
    )
}

# -p opt specification (train, evaluate and separate).
OPT_PARAMS = {
    'dest': 'configuration',
    'default': 'spleeter:2stems',
    'type': str,
    'action': 'store',
    'help': 'JSON filename that contains params'
}

# -s opt specification (separate).
OPT_OFFSET = {
    'dest': 'offset',
    'type': float,
    'default': 0.,
    'help': 'Set the starting offset to separate audio from.'
}

# -d opt specification (separate).
OPT_DURATION = {
    'dest': 'duration',
    'type': float,
    'default': 600.,
    'help': (
        'Set a maximum duration for processing audio '
        '(only separate offset + duration first seconds of '
        'the input file)')
}

# -w opt specification (separate)
OPT_STFT_BACKEND = {
    'dest': 'stft_backend',
    'type': str,
    'choices' : ["tensorflow", "librosa", "auto"],
    'default': "auto",
    'help': 'Who should be in charge of computing the stfts. Librosa is faster than tensorflow on CPU and uses'
            ' less memory. "auto" will use tensorflow when GPU acceleration is available and librosa when not.'
}


# -c opt specification (separate).
OPT_CODEC = {
    'dest': 'codec',
    'choices': ('wav', 'mp3', 'ogg', 'm4a', 'wma', 'flac'),
    'default': 'wav',
    'help': 'Audio codec to be used for the separated output'
}

# -b opt specification (separate).
OPT_BITRATE = {
    'dest': 'bitrate',
    'default': '128k',
    'help': 'Audio bitrate to be used for the separated output'
}

# -m opt specification (evaluate and separate).
OPT_MWF = {
    'dest': 'MWF',
    'action': 'store_const',
    'const': True,
    'default': False,
    'help': 'Whether to use multichannel Wiener filtering for separation',
}

# --mus_dir opt specification (evaluate).
OPT_MUSDB = {
    'dest': 'mus_dir',
    'type': str,
    'required': True,
    'help': 'Path to folder with musDB'
}

# -d opt specification (train).
OPT_DATA = {
    'dest': 'audio_path',
    'type': str,
    'required': True,
    'help': 'Path of the folder containing audio data for training'
}

# -a opt specification (train, evaluate and separate).
OPT_ADAPTER = {
    'dest': 'audio_adapter',
    'type': str,
    'help': 'Name of the audio adapter to use for audio I/O'
}

# -a opt specification (train, evaluate and separate).
OPT_VERBOSE = {
    'action': 'store_true',
    'help': 'Shows verbose logs'
}


def _add_common_options(parser):
    """ Add common option to the given parser.

    :param parser: Parser to add common opt to.
    """
    parser.add_argument('-a', '--adapter', **OPT_ADAPTER)
    parser.add_argument('-p', '--params_filename', **OPT_PARAMS)
    parser.add_argument('--verbose', **OPT_VERBOSE)


def _create_train_parser(parser_factory):
    """ Creates an argparser for training command

    :param parser_factory: Factory to use to create parser instance.
    :returns: Created and configured parser.
    """
    parser = parser_factory('train', help='Train a source separation model')
    _add_common_options(parser)
    parser.add_argument('-d', '--data', **OPT_DATA)
    return parser


def _create_evaluate_parser(parser_factory):
    """ Creates an argparser for evaluation command

    :param parser_factory: Factory to use to create parser instance.
    :returns: Created and configured parser.
    """
    parser = parser_factory(
        'evaluate',
        help='Evaluate a model on the musDB test dataset')
    _add_common_options(parser)
    parser.add_argument('-o', '--output_path', **OPT_OUTPUT)
    parser.add_argument('--mus_dir', **OPT_MUSDB)
    parser.add_argument('-m', '--mwf', **OPT_MWF)
    return parser


def _create_separate_parser(parser_factory):
    """ Creates an argparser for separation command

    :param parser_factory: Factory to use to create parser instance.
    :returns: Created and configured parser.
    """
    parser = parser_factory('separate', help='Separate audio files')
    _add_common_options(parser)
    parser.add_argument('-i', '--inputs', **OPT_INPUT)
    parser.add_argument('-o', '--output_path', **OPT_OUTPUT)
    parser.add_argument('-f', '--filename_format', **OPT_FORMAT)
    parser.add_argument('-d', '--duration', **OPT_DURATION)
    parser.add_argument('-s', '--offset', **OPT_OFFSET)
    parser.add_argument('-c', '--codec', **OPT_CODEC)
    parser.add_argument('-b', '--birate', **OPT_BITRATE)
    parser.add_argument('-m', '--mwf', **OPT_MWF)
    parser.add_argument('-B', '--stft-backend', **OPT_STFT_BACKEND)
    return parser


def create_argument_parser():
    """ Creates overall command line parser for Spleeter.

    :returns: Created argument parser.
    """
    parser = ArgumentParser(prog='spleeter')
    subparsers = parser.add_subparsers()
    subparsers.dest = 'command'
    subparsers.required = True
    _create_separate_parser(subparsers.add_parser)
    _create_train_parser(subparsers.add_parser)
    _create_evaluate_parser(subparsers.add_parser)
    return parser
