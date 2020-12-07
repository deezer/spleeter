#!/usr/bin/env python
# coding: utf8

""" TO DOCUMENT """

from pathlib import Path
from os.path import join
from spleeter.separator import STFTBackend
from tempfile import gettempdir
from typing import List

from .audio import Codec
from .audio.adapter import AudioAdapter
from .separator import Separator

# pyright: reportMissingImports=false
# pylint: disable=import-error
from typer import Argument, Option, Typer
from typer.models import OptionInfo
# pylint: enable=import-error

spleeter: Typer = Typer()
""" """

AudioOutput: OptionInfo = Option(
    join(gettempdir(), 'separated_audio'),
    help='Path of the output directory to write audio files in')

AudioSTFTBackend: OptionInfo = Option(
    STFTBackend.AUTO,
    '--stft-backend',
    '-B',
    case_sensitive=False,
    help=(
        'Who should be in charge of computing the stfts. Librosa is faster '
        'than tensorflow on CPU and uses  less memory. "auto" will use '
        'tensorflow when GPU acceleration is available and librosa when not'))

AudioAdapterDescriptor: OptionInfo = Option(
    'spleeter.audio.ffmpeg.FFMPEGProcessAudioAdapter',
    help='Name of the audio adapter to use for audio I/O')

MWF: OptionInfo = Option(
    False,
    '--mwf',
    help='Whether to use multichannel Wiener filtering for separation')

ModelParameters: OptionInfo = Option(
    'spleeter:2stems',
    help='JSON filename that contains params')

Verbose: OptionInfo = Option(
    False,
    '--verbose',
    help='Enable verbose logs')


@spleeter.command()
def train(
        adapter=None,
        verbose: bool = Verbose,
        params_filename: str = ModelParameters,
        data: Path = Option(
            ...,
            exists=True,
            dir_okay=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
            help='Path of the folder containing audio data for training')
        ) -> None:
    """
        Train a source separation model
    """
    pass


@spleeter.command()
def evaluate(
        adapter: str = AudioAdapterDescriptor,
        output_path: Path = AudioOutput,
        stft_backend: STFTBackend = AudioSTFTBackend,
        params_filename: str = ModelParameters,
        mwf: bool = MWF,
        verbose: bool = Verbose,
        mus_dir: Path = Option(
            ...,
            '--mus_dir',
            exists=True,
            dir_okay=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
            help='Path to musDB dataset directory')
        ) -> None:
    """
        Evaluate a model on the musDB test dataset
    """
    pass


@spleeter.commmand()
def separate(
        adapter: str = AudioAdapterDescriptor,
        output_path: Path = AudioOutput,
        stft_backend: STFTBackend = AudioSTFTBackend,
        params_filename: str = ModelParameters,
        mwf: bool = MWF,
        verbose: bool = Verbose,
        files: List[Path] = Argument(
            ...,
            help='List of input audio file path',
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True),
        filename_format: str = Option(
            '{filename}/{instrument}.{codec}',
            help=(
                'Template string that will be formatted to generated'
                'output filename. Such template should be Python formattable'
                'string, and could use {filename}, {instrument}, and {codec}'
                'variables')),
        duration: float = Option(
            600.,
            help=(
                'Set a maximum duration for processing audio '
                '(only separate offset + duration first seconds of '
                'the input file)')),
        offset: float = Option(
            0.,
            '--offset',
            '-s',
            help='Set the starting offset to separate audio from'),
        codec: Codec = Option(
            Codec.WAV,
            help='Audio codec to be used for the separated output'),
        bitrate: str = Option(
            '128k',
            help='Audio bitrate to be used for the separated output')
        ) -> None:
    """
        Separate audio file(s)
    """
    # TODO: try / catch or custom decorator for function handling.
    # TODO: enable_logging()
    # TODO: handle MWF
    if verbose:
        # TODO: enable_tensorflow_logging()
        pass
    # PREV: params = load_configuration(arguments.configuration)
    audio_adapter: AudioAdapter = AudioAdapter.get(adapter)
    separator: Separator = Separator(
        params_filename,
        MWF=MWF,
        stft_backend=stft_backend)
    for filename in files:
        separator.separate_to_file(
            filename,
            output_path,
            audio_adapter=audio_adapter,
            offset=offset,
            duration=duration,
            codec=codec,
            bitrate=bitrate,
            filename_format=filename_format,
            synchronous=False)
    separator.join()


if __name__ == '__main__':
    # TODO: warnings.filterwarnings('ignore')
    spleeter()
