#!/usr/bin/env python
# coding: utf8

""" This modules provides spleeter command as well as CLI parsing methods. """

from tempfile import gettempdir
from os.path import join

from ..separator import STFTBackend
from ..audio import Codec

from typer import Argument, Option
from typer.models import ArgumentInfo, OptionInfo

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

AudioInput: ArgumentInfo = Argument(
    ...,
    help='List of input audio file path',
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True)

AudioOutput: OptionInfo = Option(
    join(gettempdir(), 'separated_audio'),
    '--output_path',
    '-o',
    help='Path of the output directory to write audio files in')

AudioOffset: OptionInfo = Option(
    0.,
    '--offset',
    '-s',
    help='Set the starting offset to separate audio from')

AudioDuration: OptionInfo = Option(
    600.,
    '--duration',
    '-d',
    help=(
        'Set a maximum duration for processing audio '
        '(only separate offset + duration first seconds of '
        'the input file)'))

FilenameFormat: OptionInfo = Option(
    '{filename}/{instrument}.{codec}',
    '--filename_format',
    '-f',
    help=(
        'Template string that will be formatted to generated'
        'output filename. Such template should be Python formattable'
        'string, and could use {filename}, {instrument}, and {codec}'
        'variables'))

ModelParameters: OptionInfo = Option(
    'spleeter:2stems',
    '--params_filename',
    '-p',
    help='JSON filename that contains params')


AudioSTFTBackend: OptionInfo = Option(
    STFTBackend.AUTO,
    '--stft-backend',
    '-B',
    case_sensitive=False,
    help=(
        'Who should be in charge of computing the stfts. Librosa is faster '
        'than tensorflow on CPU and uses  less memory. "auto" will use '
        'tensorflow when GPU acceleration is available and librosa when not'))

AudioCodec: OptionInfo = Option(
    Codec.WAV,
    '--codec',
    '-c',
    help='Audio codec to be used for the separated output')

AudioBitrate: OptionInfo = Option(
    '128k',
    '--bitrate',
    '-b',
    help='Audio bitrate to be used for the separated output')

MWF: OptionInfo = Option(
    False,
    '--mwf',
    help='Whether to use multichannel Wiener filtering for separation')

MUSDBDirectory: OptionInfo = Option(
    ...,
    '--mus_dir',
    exists=True,
    dir_okay=True,
    file_okay=False,
    readable=True,
    resolve_path=True,
    help='Path to musDB dataset directory')

TrainingDataDirectory: OptionInfo = Option(
    ...,
    '--data',
    '-d',
    exists=True,
    dir_okay=True,
    file_okay=False,
    readable=True,
    resolve_path=True,
    help='Path of the folder containing audio data for training')

AudioAdapter: OptionInfo = Option(
    'spleeter.audio.ffmpeg.FFMPEGProcessAudioAdapter',
    '--adapter',
    '-a',
    help='Name of the audio adapter to use for audio I/O')

Verbose: OptionInfo = Option(
    False,
    '--verbose',
    help='Enable verbose logs')


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
    parser.add_argument('-B', '--stft-backend', **OPT_STFT_BACKEND)
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
