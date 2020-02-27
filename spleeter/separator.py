#!/usr/bin/env python
# coding: utf8

"""
    Module that provides a class wrapper for source separation.

    :Example:

    >>> from spleeter.separator import Separator
    >>> separator = Separator('spleeter:2stems')
    >>> separator.separate(waveform, lambda instrument, data: ...)
    >>> separator.separate_to_file(...)
"""

import os
import logging

from time import time
from multiprocessing import Pool
from os.path import basename, join, splitext
import numpy as np
import tensorflow as tf
from librosa.core import stft, istft
from scipy.signal.windows import hann

from . import SpleeterError
from .audio.adapter import get_default_audio_adapter
from .audio.convertor import to_stereo
from .utils.configuration import load_configuration
from .utils.estimator import create_estimator, to_predictor, get_default_model_dir
from .model import EstimatorSpecBuilder, InputProviderFactory


__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


logger = logging.getLogger("spleeter")



def get_backend(backend):
    assert backend in ["auto", "tensorflow", "librosa"]
    if backend == "auto":
        return "tensorflow" if tf.test.is_gpu_available() else "librosa"
    return backend


class Separator(object):
    """ A wrapper class for performing separation. """

    def __init__(self, params_descriptor, MWF=False, stft_backend="auto", multiprocess=True):
        """ Default constructor.

        :param params_descriptor: Descriptor for TF params to be used.
        :param MWF: (Optional) True if MWF should be used, False otherwise.
        """

        self._params = load_configuration(params_descriptor)
        self._sample_rate = self._params['sample_rate']
        self._MWF = MWF
        self._predictor = None
        self._pool = Pool() if multiprocess else None
        self._tasks = []
        self._params["stft_backend"] = get_backend(stft_backend)

    def _get_predictor(self):
        """ Lazy loading access method for internal predictor instance.

        :returns: Predictor to use for source separation.
        """
        if self._predictor is None:
            estimator = create_estimator(self._params, self._MWF)
            self._predictor = to_predictor(estimator)
        return self._predictor

    def join(self, timeout=200):
        """ Wait for all pending tasks to be finished.

        :param timeout: (Optional) task waiting timeout.
        """
        while len(self._tasks) > 0:
            task = self._tasks.pop()
            task.get()
            task.wait(timeout=timeout)

    def separate_tensorflow(self, waveform, audio_descriptor):
        """ Performs source separation over the given waveform.

        The separation is performed synchronously but the result
        processing is done asynchronously, allowing for instance
        to export audio in parallel (through multiprocessing).

        Given result is passed by to the given consumer, which will
        be waited for task finishing if synchronous flag is True.

        :param waveform: Waveform to apply separation on.
        :returns: Separated waveforms.
        """
        if not waveform.shape[-1] == 2:
            waveform = to_stereo(waveform)
        predictor = self._get_predictor()
        prediction = predictor({
            'waveform': waveform,
            'audio_id': audio_descriptor})
        prediction.pop('audio_id')
        return prediction

    def stft(self, data, inverse=False, length=None):
        """
        Single entrypoint for both stft and istft. This computes stft and istft with librosa on stereo data. The two
        channels are processed separately and are concatenated together in the result. The expected input formats are:
        (n_samples, 2) for stft and (T, F, 2) for istft.
        :param data: np.array with either the waveform or the complex spectrogram depending on the parameter inverse
        :param inverse: should a stft or an istft be computed.
        :return: Stereo data as numpy array for the transform. The channels are stored in the last dimension
        """
        assert not (inverse and length is None)
        data = np.asfortranarray(data)
        N = self._params["frame_length"]
        H = self._params["frame_step"]
        win = hann(N, sym=False)
        fstft = istft if inverse else stft
        win_len_arg = {"win_length": None, "length": length} if inverse else {"n_fft": N}
        dl, dr = (data[:, :, 0].T, data[:, :, 1].T) if inverse else (data[:, 0], data[:, 1])
        s1 = fstft(dl, hop_length=H, window=win, center=False, **win_len_arg)
        s2 = fstft(dr, hop_length=H, window=win, center=False, **win_len_arg)
        s1 = np.expand_dims(s1.T, 2-inverse)
        s2 = np.expand_dims(s2.T, 2-inverse)
        return np.concatenate([s1, s2], axis=2-inverse)

    def separate_librosa(self, waveform, audio_id):
        out = {}
        input_provider = InputProviderFactory.get(self._params)
        features = input_provider.get_input_dict_placeholders()

        builder = EstimatorSpecBuilder(features, self._params)
        latest_checkpoint = tf.train.latest_checkpoint(get_default_model_dir(self._params['model_dir']))

        # TODO: fix the logic, build sometimes return, sometimes set attribute
        outputs = builder.outputs

        saver = tf.train.Saver()
        stft = self.stft(waveform)
        with tf.Session() as sess:
            saver.restore(sess, latest_checkpoint)
            outputs = sess.run(outputs, feed_dict=input_provider.get_feed_dict(features, stft, audio_id))
            for inst in builder.instruments:
                out[inst] = self.stft(outputs[inst], inverse=True, length=waveform.shape[0])
        return out

    def separate(self, waveform, audio_descriptor):
        if self._params["stft_backend"] == "tensorflow":
            return self.separate_tensorflow(waveform, audio_descriptor)
        else:
            return self.separate_librosa(waveform, audio_descriptor)

    def separate_to_file(
            self, audio_descriptor, destination,
            audio_adapter=get_default_audio_adapter(),
            offset=0, duration=600., codec='wav', bitrate='128k',
            filename_format='{filename}/{instrument}.{codec}',
            synchronous=True):
        """ Performs source separation and export result to file using
        given audio adapter.

        Filename format should be a Python formattable string that could use
        following parameters : {instrument}, {filename} and {codec}.

        :param audio_descriptor:    Describe song to separate, used by audio
                                    adapter to retrieve and load audio data,
                                    in case of file based audio adapter, such
                                    descriptor would be a file path.
        :param destination:         Target directory to write output to.
        :param audio_adapter:       (Optional) Audio adapter to use for I/O.
        :param chunk_duration:      (Optional) Maximum signal duration that is processed
                                               in one pass. Default: all signal.
        :param offset:              (Optional) Offset of loaded song.
        :param duration:            (Optional) Duration of loaded song.
        :param codec:               (Optional) Export codec.
        :param bitrate:             (Optional) Export bitrate.
        :param filename_format:     (Optional) Filename format.
        :param synchronous:         (Optional) True is should by synchronous.
        """
        waveform, sample_rate = audio_adapter.load(
            audio_descriptor,
            offset=offset,
            duration=duration,
            sample_rate=self._sample_rate)
        sources = self.separate(waveform, audio_descriptor)
        self.save_to_file(sources, audio_descriptor, destination, filename_format, codec,
                          audio_adapter, bitrate, synchronous)

    def save_to_file(self, sources, audio_descriptor, destination, filename_format, codec,
                     audio_adapter, bitrate, synchronous):
        filename = splitext(basename(audio_descriptor))[0]
        generated = []
        for instrument, data in sources.items():
            path = join(destination, filename_format.format(
                filename=filename,
                instrument=instrument,
                codec=codec))
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if path in generated:
                raise SpleeterError((
                    f'Separated source path conflict : {path},'
                    'please check your filename format'))
            generated.append(path)
            if self._pool:
                task = self._pool.apply_async(audio_adapter.save, (
                    path,
                    data,
                    self._sample_rate,
                    codec,
                    bitrate))
                self._tasks.append(task)
            else:
                audio_adapter.save(path, data, self._sample_rate, codec, bitrate)
        if synchronous and self._pool:
            self.join()
