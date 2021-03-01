#!/usr/bin/env python
# coding: utf8

""" This modules provides spleeter command as well as CLI parsing methods. """

from os.path import join
from tempfile import gettempdir

from typer import Argument, Exit, Option, echo
from typer.models import ArgumentInfo, OptionInfo

from .audio import Codec, STFTBackend

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

AudioInputArgument: ArgumentInfo = Argument(
    ...,
    help="List of input audio file path",
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
)

AudioInputOption: OptionInfo = Option(
    None, "--inputs", "-i", help="(DEPRECATED) placeholder for deprecated input option"
)

AudioAdapterOption: OptionInfo = Option(
    "spleeter.audio.ffmpeg.FFMPEGProcessAudioAdapter",
    "--adapter",
    "-a",
    help="Name of the audio adapter to use for audio I/O",
)

AudioOutputOption: OptionInfo = Option(
    join(gettempdir(), "separated_audio"),
    "--output_path",
    "-o",
    help="Path of the output directory to write audio files in",
)

AudioOffsetOption: OptionInfo = Option(
    0.0, "--offset", "-s", help="Set the starting offset to separate audio from"
)

AudioDurationOption: OptionInfo = Option(
    600.0,
    "--duration",
    "-d",
    help=(
        "Set a maximum duration for processing audio "
        "(only separate offset + duration first seconds of "
        "the input file)"
    ),
)

AudioSTFTBackendOption: OptionInfo = Option(
    STFTBackend.AUTO,
    "--stft-backend",
    "-B",
    case_sensitive=False,
    help=(
        "Who should be in charge of computing the stfts. Librosa is faster "
        'than tensorflow on CPU and uses  less memory. "auto" will use '
        "tensorflow when GPU acceleration is available and librosa when not"
    ),
)

AudioCodecOption: OptionInfo = Option(
    Codec.WAV, "--codec", "-c", help="Audio codec to be used for the separated output"
)

AudioBitrateOption: OptionInfo = Option(
    "128k", "--bitrate", "-b", help="Audio bitrate to be used for the separated output"
)

FilenameFormatOption: OptionInfo = Option(
    "{filename}/{instrument}.{codec}",
    "--filename_format",
    "-f",
    help=(
        "Template string that will be formatted to generated"
        "output filename. Such template should be Python formattable"
        "string, and could use {filename}, {instrument}, and {codec}"
        "variables"
    ),
)

ModelParametersOption: OptionInfo = Option(
    "spleeter:2stems",
    "--params_filename",
    "-p",
    help="JSON filename that contains params",
)


MWFOption: OptionInfo = Option(
    False, "--mwf", help="Whether to use multichannel Wiener filtering for separation"
)

MUSDBDirectoryOption: OptionInfo = Option(
    ...,
    "--mus_dir",
    exists=True,
    dir_okay=True,
    file_okay=False,
    readable=True,
    resolve_path=True,
    help="Path to musDB dataset directory",
)

TrainingDataDirectoryOption: OptionInfo = Option(
    ...,
    "--data",
    "-d",
    exists=True,
    dir_okay=True,
    file_okay=False,
    readable=True,
    resolve_path=True,
    help="Path of the folder containing audio data for training",
)

VerboseOption: OptionInfo = Option(False, "--verbose", help="Enable verbose logs")


def version_callback(value: bool):
    if value:
        try:
            from importlib.metadata import version
        except ImportError:
            from importlib_metadata import version

        echo(f"Spleeter Version: {version('spleeter')}")
        raise Exit()


VersionOption: OptionInfo = Option(
    None,
    "--version",
    callback=version_callback,
    is_eager=True,
    help="Return Spleeter version",
)
