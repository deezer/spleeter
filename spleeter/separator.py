#!/usr/bin/env python
# coding: utf8

"""
    Module that provides a class wrapper for source separation.

    Examples:

    ```python
    >>> from spleeter.separator import Separator
    >>> separator = Separator('spleeter:2stems')
    >>> separator.separate(waveform, lambda instrument, data: ...)
    >>> separator.separate_to_file(...)
    ```
"""

import atexit
import os
from multiprocessing import Pool
from os.path import basename, dirname, join, splitext
from typing import Dict, Generator, Optional

# pyright: reportMissingImports=false
# pylint: disable=import-error
import numpy as np
import tensorflow as tf
from librosa.core import istft, stft
from scipy.signal.windows import hann

from spleeter.model.provider import ModelProvider

from . import SpleeterError
from .audio import Codec, STFTBackend
from .audio.adapter import AudioAdapter
from .audio.convertor import to_stereo
from .model import EstimatorSpecBuilder, InputProviderFactory, model_fn
from .model.provider import ModelProvider
from .types import AudioDescriptor
from .utils.configuration import load_configuration

# pylint: enable=import-error

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


class DataGenerator(object):
    """
    Generator object that store a sample and generate it once while called.
    Used to feed a tensorflow estimator without knowing the whole data at
    build time.
    """

    def __init__(self) -> None:
        """Default constructor."""
        self._current_data = None

    def update_data(self, data) -> None:
        """Replace internal data."""
        self._current_data = data

    def __call__(self) -> Generator:
        """Generation process."""
        buffer = self._current_data
        while buffer:
            yield buffer
            buffer = self._current_data


def create_estimator(params, MWF):
    """
    Initialize tensorflow estimator that will perform separation

    Params:
    - params: a dictionary of parameters for building the model

    Returns:
        a tensorflow estimator
    """
    # Load model.
    provider: ModelProvider = ModelProvider.default()
    params["model_dir"] = provider.get(params["model_dir"])
    params["MWF"] = MWF
    # Setup config
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config = tf.estimator.RunConfig(session_config=session_config)
    # Setup estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=params["model_dir"], params=params, config=config
    )
    return estimator


class Separator(object):
    """A wrapper class for performing separation."""

    def __init__(
        self,
        params_descriptor: str,
        MWF: bool = False,
        stft_backend: STFTBackend = STFTBackend.AUTO,
        multiprocess: bool = True,
    ) -> None:
        """
        Default constructor.

        Parameters:
            params_descriptor (str):
                Descriptor for TF params to be used.
            MWF (bool):
                (Optional) `True` if MWF should be used, `False` otherwise.
        """
        self._params = load_configuration(params_descriptor)
        self._sample_rate = self._params["sample_rate"]
        self._MWF = MWF
        self._tf_graph = tf.Graph()
        self._prediction_generator = None
        self._input_provider = None
        self._builder = None
        self._features = None
        self._session = None
        if multiprocess:
            self._pool = Pool()
            atexit.register(self._pool.close)
        else:
            self._pool = None
        self._tasks = []
        self._params["stft_backend"] = STFTBackend.resolve(stft_backend)
        self._data_generator = DataGenerator()

    def _get_prediction_generator(self) -> Generator:
        """
        Lazy loading access method for internal prediction generator
        returned by the predict method of a tensorflow estimator.

        Returns:
            Generator:
                Generator of prediction.
        """
        if self._prediction_generator is None:
            estimator = create_estimator(self._params, self._MWF)

            def get_dataset():
                return tf.data.Dataset.from_generator(
                    self._data_generator,
                    output_types={"waveform": tf.float32, "audio_id": tf.string},
                    output_shapes={"waveform": (None, 2), "audio_id": ()},
                )

            self._prediction_generator = estimator.predict(
                get_dataset, yield_single_examples=False
            )
        return self._prediction_generator

    def join(self, timeout: int = 200) -> None:
        """
        Wait for all pending tasks to be finished.

        Parameters:
            timeout (int):
                (Optional) task waiting timeout.
        """
        while len(self._tasks) > 0:
            task = self._tasks.pop()
            task.get()
            task.wait(timeout=timeout)

    def _stft(
        self, data: np.ndarray, inverse: bool = False, length: Optional[int] = None
    ) -> np.ndarray:
        """
        Single entrypoint for both stft and istft. This computes stft and
        istft with librosa on stereo data. The two channels are processed
        separately and are concatenated together in the result. The
        expected input formats are: (n_samples, 2) for stft and (T, F, 2)
        for istft.

        Parameters:
            data (numpy.array):
                Array with either the waveform or the complex spectrogram
                depending on the parameter inverse
            inverse (bool):
                (Optional) Should a stft or an istft be computed.
            length (Optional[int]):

        Returns:
            numpy.ndarray:
                Stereo data as numpy array for the transform. The channels
                are stored in the last dimension.
        """
        assert not (inverse and length is None)
        data = np.asfortranarray(data)
        N = self._params["frame_length"]
        H = self._params["frame_step"]
        win = hann(N, sym=False)
        fstft = istft if inverse else stft
        win_len_arg = {"win_length": None, "length": None} if inverse else {"n_fft": N}
        n_channels = data.shape[-1]
        out = []
        for c in range(n_channels):
            d = (
                np.concatenate((np.zeros((N,)), data[:, c], np.zeros((N,))))
                if not inverse
                else data[:, :, c].T
            )
            s = fstft(d, hop_length=H, window=win, center=False, **win_len_arg)
            if inverse:
                s = s[N : N + length]
            s = np.expand_dims(s.T, 2 - inverse)
            out.append(s)
        if len(out) == 1:
            return out[0]
        return np.concatenate(out, axis=2 - inverse)

    def _get_input_provider(self):
        if self._input_provider is None:
            self._input_provider = InputProviderFactory.get(self._params)
        return self._input_provider

    def _get_features(self):
        if self._features is None:
            provider = self._get_input_provider()
            self._features = provider.get_input_dict_placeholders()
        return self._features

    def _get_builder(self):
        if self._builder is None:
            self._builder = EstimatorSpecBuilder(self._get_features(), self._params)
        return self._builder

    def _get_session(self):
        if self._session is None:
            saver = tf.compat.v1.train.Saver()
            provider = ModelProvider.default()
            model_directory: str = provider.get(self._params["model_dir"])
            latest_checkpoint = tf.train.latest_checkpoint(model_directory)
            self._session = tf.compat.v1.Session()
            saver.restore(self._session, latest_checkpoint)
        return self._session

    def _separate_librosa(
        self, waveform: np.ndarray, audio_descriptor: AudioDescriptor
    ) -> Dict:
        """
        Performs separation with librosa backend for STFT.

        Parameters:
            waveform (numpy.ndarray):
                Waveform to be separated (as a numpy array)
            audio_descriptor (AudioDescriptor):
        """
        with self._tf_graph.as_default():
            out = {}
            features = self._get_features()
            # TODO: fix the logic, build sometimes return,
            #       sometimes set attribute.
            outputs = self._get_builder().outputs
            stft = self._stft(waveform)
            if stft.shape[-1] == 1:
                stft = np.concatenate([stft, stft], axis=-1)
            elif stft.shape[-1] > 2:
                stft = stft[:, :2]
            sess = self._get_session()
            outputs = sess.run(
                outputs,
                feed_dict=self._get_input_provider().get_feed_dict(
                    features, stft, audio_descriptor
                ),
            )
            for inst in self._get_builder().instruments:
                out[inst] = self._stft(
                    outputs[inst], inverse=True, length=waveform.shape[0]
                )
            return out

    def _separate_tensorflow(
        self, waveform: np.ndarray, audio_descriptor: AudioDescriptor
    ) -> Dict:
        """
        Performs source separation over the given waveform with tensorflow
        backend.

        Parameters:
            waveform (numpy.ndarray):
                Waveform to be separated (as a numpy array)
            audio_descriptor (AudioDescriptor):

        Returns:
            Separated waveforms.
        """
        if not waveform.shape[-1] == 2:
            waveform = to_stereo(waveform)
        prediction_generator = self._get_prediction_generator()
        # NOTE: update data in generator before performing separation.
        self._data_generator.update_data(
            {"waveform": waveform, "audio_id": np.array(audio_descriptor)}
        )
        # NOTE: perform separation.
        prediction = next(prediction_generator)
        prediction.pop("audio_id")
        return prediction

    def separate(
        self, waveform: np.ndarray, audio_descriptor: Optional[str] = ""
    ) -> None:
        """
        Performs separation on a waveform.

        Parameters:
            waveform (numpy.ndarray):
                Waveform to be separated (as a numpy array)
            audio_descriptor (str):
                (Optional) string describing the waveform (e.g. filename).
        """
        backend: str = self._params["stft_backend"]
        if backend == STFTBackend.TENSORFLOW:
            return self._separate_tensorflow(waveform, audio_descriptor)
        elif backend == STFTBackend.LIBROSA:
            return self._separate_librosa(waveform, audio_descriptor)
        raise ValueError(f"Unsupported STFT backend {backend}")

    def separate_to_file(
        self,
        audio_descriptor: AudioDescriptor,
        destination: str,
        audio_adapter: Optional[AudioAdapter] = None,
        offset: int = 0,
        duration: float = 600.0,
        codec: Codec = Codec.WAV,
        bitrate: str = "128k",
        filename_format: str = "{filename}/{instrument}.{codec}",
        synchronous: bool = True,
    ) -> None:
        """
        Performs source separation and export result to file using
        given audio adapter.

        Filename format should be a Python formattable string that could
        use following parameters :

        - {instrument}
        - {filename}
        - {foldername}
        - {codec}.

        Parameters:
            audio_descriptor (AudioDescriptor):
                Describe song to separate, used by audio adapter to
                retrieve and load audio data, in case of file based
                audio adapter, such descriptor would be a file path.
            destination (str):
                Target directory to write output to.
            audio_adapter (Optional[AudioAdapter]):
                (Optional) Audio adapter to use for I/O.
            offset (int):
                (Optional) Offset of loaded song.
            duration (float):
                (Optional) Duration of loaded song (default: 600s).
            codec (Codec):
                (Optional) Export codec.
            bitrate (str):
                (Optional) Export bitrate.
            filename_format (str):
                (Optional) Filename format.
            synchronous (bool):
                (Optional) True is should by synchronous.
        """
        if audio_adapter is None:
            audio_adapter = AudioAdapter.default()
        waveform, _ = audio_adapter.load(
            audio_descriptor,
            offset=offset,
            duration=duration,
            sample_rate=self._sample_rate,
        )
        sources = self.separate(waveform, audio_descriptor)
        self.save_to_file(
            sources,
            audio_descriptor,
            destination,
            filename_format,
            codec,
            audio_adapter,
            bitrate,
            synchronous,
        )

    def save_to_file(
        self,
        sources: Dict,
        audio_descriptor: AudioDescriptor,
        destination: str,
        filename_format: str = "{filename}/{instrument}.{codec}",
        codec: Codec = Codec.WAV,
        audio_adapter: Optional[AudioAdapter] = None,
        bitrate: str = "128k",
        synchronous: bool = True,
    ) -> None:
        """
        Export dictionary of sources to files.

        Parameters:
            sources (Dict):
                Dictionary of sources to be exported. The keys are the name
                of the instruments, and the values are `N x 2` numpy arrays
                containing the corresponding intrument waveform, as
                returned by the separate method
            audio_descriptor (AudioDescriptor):
                Describe song to separate, used by audio adapter to
                retrieve and load audio data, in case of file based audio
                adapter, such descriptor would be a file path.
            destination (str):
                Target directory to write output to.
            filename_format (str):
                (Optional) Filename format.
            codec (Codec):
                (Optional) Export codec.
            audio_adapter (Optional[AudioAdapter]):
                (Optional) Audio adapter to use for I/O.
            bitrate (str):
                (Optional) Export bitrate.
            synchronous (bool):
                (Optional) True is should by synchronous.
        """
        if audio_adapter is None:
            audio_adapter = AudioAdapter.default()
        foldername = basename(dirname(audio_descriptor))
        filename = splitext(basename(audio_descriptor))[0]
        generated = []
        for instrument, data in sources.items():
            path = join(
                destination,
                filename_format.format(
                    filename=filename,
                    instrument=instrument,
                    foldername=foldername,
                    codec=codec,
                ),
            )
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if path in generated:
                raise SpleeterError(
                    (
                        f"Separated source path conflict : {path},"
                        "please check your filename format"
                    )
                )
            generated.append(path)
            if self._pool:
                task = self._pool.apply_async(
                    audio_adapter.save, (path, data, self._sample_rate, codec, bitrate)
                )
                self._tasks.append(task)
            else:
                audio_adapter.save(path, data, self._sample_rate, codec, bitrate)
        if synchronous and self._pool:
            self.join()
