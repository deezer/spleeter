#!/usr/bin/env python
# coding: utf8

""" This modules provides spleeter command as well as CLI parsing methods. """

import json

from argparse import ArgumentParser
from tempfile import gettempdir
from os.path import exists, join

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

# -i opt specification.
OPT_INPUT = {
    'dest': 'audio_filenames',
    'nargs': '+',
    'help': 'List of input audio filenames',
    'required': True
}

# -o opt specification.
OPT_OUTPUT = {
    'dest': 'output_path',
    'default': join(gettempdir(), 'separated_audio'),
    'help': 'Path of the output directory to write audio files in'
}

# -p opt specification.
OPT_PARAMS = {
    'dest': 'params_filename',
    'default': 'spleeter:2stems',
    'type': str,
    'action': 'store',
    'help': 'JSON filename that contains params'
}

# -n opt specification.
OPT_OUTPUT_NAMING = {
    'dest': 'output_naming',
    'default': 'filename',
    'choices': ('directory', 'filename'),
    'help': (
        'Choice for naming the output base path: '
        '"filename" (use the input filename, i.e '
        '/path/to/audio/mix.wav will be separated to '
        '<output_path>/mix/<instument1>.wav, '
        '<output_path>/mix/<instument2>.wav...) or '
        '"directory" (use the name of the input last level'
        ' directory, for instance /path/to/audio/mix.wav '
        'will be separated to <output_path>/audio/<instument1>.wav'
        ', <output_path>/audio/<instument2>.wav)')
}

# -d opt specification (separate).
OPT_DURATION = {
    'dest': 'max_duration',
    'type': float,
    'default': 600.,
    'help': (
        'Set a maximum duration for processing audio '
        '(only separate max_duration first seconds of '
        'the input file)')
}

# -c opt specification.
OPT_CODEC = {
    'dest': 'audio_codec',
    'choices': ('wav', 'mp3', 'ogg', 'm4a', 'wma', 'flac'),
    'default': 'wav',
    'help': 'Audio codec to be used for the separated output'
}

# -m opt specification.
OPT_MWF = {
    'dest': 'MWF',
    'action': 'store_const',
    'const': True,
    'default': False,
    'help': 'Whether to use multichannel Wiener filtering for separation',
}

# --mus_dir opt specification.
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

# -a opt specification.
OPT_ADAPTER = {
    'dest': 'audio_adapter',
    'type': str,
    'help': 'Name of the audio adapter to use for audio I/O'
}

# -a opt specification.
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
    parser.add_argument('-i', '--audio_filenames', **OPT_INPUT)
    parser.add_argument('-o', '--output_path', **OPT_OUTPUT)
    parser.add_argument('-n', '--output_naming', **OPT_OUTPUT_NAMING)
    parser.add_argument('-d', '--max_duration', **OPT_DURATION)
    parser.add_argument('-c', '--audio_codec', **OPT_CODEC)
    parser.add_argument('-m', '--mwf', **OPT_MWF)
    return parser


def create_argument_parser():
    """ Creates overall command line parser for Spleeter.

    :returns: Created argument parser.
    """
    parser = ArgumentParser(prog='python -m spleeter')
    subparsers = parser.add_subparsers()
    subparsers.dest = 'command'
    subparsers.required = True
    _create_separate_parser(subparsers.add_parser)
    _create_train_parser(subparsers.add_parser)
    _create_evaluate_parser(subparsers.add_parser)
    return parser
