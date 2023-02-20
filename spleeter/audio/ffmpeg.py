#!/usr/bin/env python
# coding: utf8

"""
This module provides an AudioAdapter implementation based on FFMPEG
process. Such implementation is POSIXish and depends on nothing except
standard Python libraries. Thus this implementation is the default one
used within this library.
"""

import datetime as dt
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

# pyright: reportMissingImports=false
# pylint: disable=import-error
import ffmpeg  # type: ignore
import numpy as np

from .. import SpleeterError
from ..types import Signal
from ..utils.logging import logger
from . import Codec
from .adapter import AudioAdapter

# pylint: enable=import-error

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


class FFMPEGProcessAudioAdapter(AudioAdapter):
    """
    An AudioAdapter implementation that use FFMPEG binary through
    subprocess in order to perform I/O operation for audio processing.

    When created, FFMPEG binary path will be checked and expended,
    raising exception if not found. Such path could be infered using
    `FFMPEG_PATH` environment variable.
    """

    SUPPORTED_CODECS: Dict[Codec, str] = {
        Codec.M4A: "aac",
        Codec.OGG: "libvorbis",
        Codec.WMA: "wmav2",
    }
    """ FFMPEG codec name mapping. """

    def __init__(_) -> None:
        """
        Default constructor, ensure FFMPEG binaries are available.

        Raises:
            SpleeterError:
                If ffmpeg or ffprobe is not found.
        """
        for binary in ("ffmpeg", "ffprobe"):
            if shutil.which(binary) is None:
                raise SpleeterError("{} binary not found".format(binary))

    def load(
        _,
        path: Union[Path, str],
        offset: Optional[float] = None,
        duration: Optional[float] = None,
        sample_rate: Optional[float] = None,
        dtype: bytes = b"float32",
    ) -> Signal:
        """
        Loads the audio file denoted by the given path
        and returns it data as a waveform.

        Parameters:
            path (Union[Path, str]:
                Path of the audio file to load data from.
            offset (Optional[float]):
                (Optional) Start offset to load from in seconds.
            duration (Optional[float]):
                (Optional) Duration to load in seconds.
            sample_rate (Optional[float]):
                (Optional) Sample rate to load audio with.
            dtype (bytes):
                (Optional) Data type to use, default to `b'float32'`.

        Returns:
            Signal:
                Loaded data a (waveform, sample_rate) tuple.

        Raises:
            SpleeterError:
                If any error occurs while loading audio.
        """
        if isinstance(path, Path):
            path = str(path)
        if not isinstance(path, str):
            path = path.decode()
        try:
            probe = ffmpeg.probe(path)
        except ffmpeg._run.Error as e:
            raise SpleeterError(
                "An error occurs with ffprobe (see ffprobe output below)\n\n{}".format(
                    e.stderr.decode()
                )
            )
        if "streams" not in probe or len(probe["streams"]) == 0:
            raise SpleeterError("No stream was found with ffprobe")
        metadata = next(
            stream for stream in probe["streams"] if stream["codec_type"] == "audio"
        )
        n_channels = metadata["channels"]
        if sample_rate is None:
            sample_rate = metadata["sample_rate"]
        output_kwargs = {"format": "f32le", "ar": sample_rate}
        if duration is not None:
            output_kwargs["t"] = str(dt.timedelta(seconds=duration))
        if offset is not None:
            output_kwargs["ss"] = str(dt.timedelta(seconds=offset))
        process = (
            ffmpeg.input(path)
            .output("pipe:", **output_kwargs)
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        buffer, _ = process.communicate()
        waveform = np.frombuffer(buffer, dtype="<f4").reshape(-1, n_channels)
        if not waveform.dtype == np.dtype(dtype):
            waveform = waveform.astype(dtype)
        return (waveform, sample_rate)

    def save(
        self,
        path: Union[Path, str],
        data: np.ndarray,
        sample_rate: float,
        codec: Codec = None,
        bitrate: str = None,
    ) -> None:
        """
        Write waveform data to the file denoted by the given path using
        FFMPEG process.

        Parameters:
            path (Union[Path, str]):
                Path like of the audio file to save data in.
            data (np.ndarray):
                Waveform data to write.
            sample_rate (float):
                Sample rate to write file in.
            codec (Codec):
                (Optional) Writing codec to use, default to `None`.
            bitrate (str):
                (Optional) Bitrate of the written audio file, default to
                `None`.

        Raises:
            IOError:
                If any error occurs while using FFMPEG to write data.
        """
        if isinstance(path, Path):
            path = str(path)
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            raise SpleeterError(f"output directory does not exists: {directory}")
        logger.debug(f"Writing file {path}")
        input_kwargs = {"ar": sample_rate, "ac": data.shape[1]}
        output_kwargs = {"ar": sample_rate, "strict": "-2"}
        if bitrate:
            output_kwargs["audio_bitrate"] = bitrate
        if codec is not None and codec != "wav":
            output_kwargs["codec"] = self.SUPPORTED_CODECS.get(codec, codec)
        process = (
            ffmpeg.input("pipe:", format="f32le", **input_kwargs)
            .output(path, **output_kwargs)
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True)
        )
        try:
            process.stdin.write(data.astype("<f4").tobytes())
            process.stdin.close()
            process.wait()
        except IOError:
            raise SpleeterError(f"FFMPEG error: {process.stderr.read()}")
        logger.info(f"File {path} written succesfully")
