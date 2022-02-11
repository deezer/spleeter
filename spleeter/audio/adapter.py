#!/usr/bin/env python
# coding: utf8

""" AudioAdapter class defintion. """

from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# pyright: reportMissingImports=false
# pylint: disable=import-error
import numpy as np
import tensorflow as tf

from spleeter.audio import Codec

from .. import SpleeterError
from ..types import AudioDescriptor, Signal
from ..utils.logging import logger

# pylint: enable=import-error


__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


class AudioAdapter(ABC):
    """An abstract class for manipulating audio signal."""

    _DEFAULT: "AudioAdapter" = None
    """ Default audio adapter singleton instance. """

    @abstractmethod
    def load(
        self,
        audio_descriptor: AudioDescriptor,
        offset: Optional[float] = None,
        duration: Optional[float] = None,
        sample_rate: Optional[float] = None,
        dtype: np.dtype = np.float32,
    ) -> Signal:
        """
        Loads the audio file denoted by the given audio descriptor and
        returns it data as a waveform. Aims to be implemented by client.

        Parameters:
            audio_descriptor (AudioDescriptor):
                Describe song to load, in case of file based audio adapter,
                such descriptor would be a file path.
            offset (Optional[float]):
                Start offset to load from in seconds.
            duration (Optional[float]):
                Duration to load in seconds.
            sample_rate (Optional[float]):
                Sample rate to load audio with.
            dtype (numpy.dtype):
                (Optional) Numpy data type to use, default to `float32`.

        Returns:
            Signal:
                Loaded data as (wf, sample_rate) tuple.
        """
        pass

    def load_tf_waveform(
        self,
        audio_descriptor,
        offset: float = 0.0,
        duration: float = 1800.0,
        sample_rate: int = 44100,
        dtype: bytes = b"float32",
        waveform_name: str = "waveform",
    ) -> Dict[str, Any]:
        """
        Load the audio and convert it to a tensorflow waveform.

        Parameters:
            audio_descriptor ():
                Describe song to load, in case of file based audio adapter,
                such descriptor would be a file path.
            offset (float):
                Start offset to load from in seconds.
            duration (float):
                Duration to load in seconds.
            sample_rate (float):
                Sample rate to load audio with.
            dtype (bytes):
                (Optional)data type to use, default to `b'float32'`.
            waveform_name (str):
                (Optional) Name of the key in output dict, default to
                `'waveform'`.

        Returns:
            Dict[str, Any]:
                TF output dict with waveform as `(T x chan numpy array)`
                and a boolean that tells whether there were an error while
                trying to load the waveform.
        """
        # Cast parameters to TF format.
        offset = tf.cast(offset, tf.float64)
        duration = tf.cast(duration, tf.float64)

        # Defined safe loading function.
        def safe_load(path, offset, duration, sample_rate, dtype):
            logger.info(f"Loading audio {path} from {offset} to {offset + duration}")
            try:
                (data, _) = self.load(
                    path.numpy(),
                    offset.numpy(),
                    duration.numpy(),
                    sample_rate.numpy(),
                    dtype=dtype.numpy(),
                )
                logger.info("Audio data loaded successfully")
                return (data, False)
            except Exception as e:
                logger.exception("An error occurs while loading audio", exc_info=e)
            return (np.float32(-1.0), True)

        # Execute function and format results.
        results = (
            tf.py_function(
                safe_load,
                [audio_descriptor, offset, duration, sample_rate, dtype],
                (tf.float32, tf.bool),
            ),
        )
        waveform, error = results[0]
        return {waveform_name: waveform, f"{waveform_name}_error": error}

    @abstractmethod
    def save(
        self,
        path: Union[Path, str],
        data: np.ndarray,
        sample_rate: float,
        codec: Codec = None,
        bitrate: str = None,
    ) -> None:
        """
        Save the given audio data to the file denoted by the given path.

        Parameters:
            path (Union[Path, str]):
                Path like of the audio file to save data in.
            data (numpy.ndarray):
                Waveform data to write.
            sample_rate (float):
                Sample rate to write file in.
            codec ():
                (Optional) Writing codec to use, default to `None`.
            bitrate (str):
                (Optional) Bitrate of the written audio file, default to
                `None`.
        """
        pass

    @classmethod
    def default(cls: type) -> "AudioAdapter":
        """
        Builds and returns a default audio adapter instance.

        Returns:
            AudioAdapter:
                Default adapter instance to use.
        """
        if cls._DEFAULT is None:
            from .ffmpeg import FFMPEGProcessAudioAdapter

            cls._DEFAULT = FFMPEGProcessAudioAdapter()
        return cls._DEFAULT

    @classmethod
    def get(cls: type, descriptor: str) -> "AudioAdapter":
        """
        Load dynamically an AudioAdapter from given class descriptor.

        Parameters:
            descriptor (str):
                Adapter class descriptor (module.Class)

        Returns:
            AudioAdapter:
                Created adapter instance.
        """
        if not descriptor:
            return cls.default()
        module_path: List[str] = descriptor.split(".")
        adapter_class_name: str = module_path[-1]
        module_path: str = ".".join(module_path[:-1])
        adapter_module = import_module(module_path)
        adapter_class = getattr(adapter_module, adapter_class_name)
        if not issubclass(adapter_class, AudioAdapter):
            raise SpleeterError(
                f"{adapter_class_name} is not a valid AudioAdapter class"
            )
        return adapter_class()
