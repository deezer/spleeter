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
        self._params["stft_backend"] = stft_backend

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

    def separate(self, waveform):
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
            'audio_id': ''})
        prediction.pop('audio_id')
        return prediction

    def get_valid_chunk_size(self, sample_rate: int, chunk_max_duration: float) -> int:
        """
        Given a sample rate, and a maximal duration that a chunk can represent, return the maximum chunk
        size in samples. The chunk size must be a non-zero multiple of T (temporal dimension of the input spectrogram)
        times F (number of frequency bins in the input spectrogram). If no such value exist, we return T*F.

        :param sample_rate:         sample rate of the pcm data
        :param chunk_max_duration:  maximal duration in seconds of a chunk
        :return: highest non-zero chunk size of duration less than chunk_max_duration or minimal valid chunk size.
        """
        assert chunk_max_duration > 0
        chunk_size = chunk_max_duration * sample_rate
        min_sample_size = self._params["T"] * self._params["F"]
        if chunk_size < min_sample_size:
            min_duration = min_sample_size / sample_rate
            logger.warning("chunk_duration must be at least {:.2f} seconds. Ignoring parameter".format(min_duration))
            chunk_size = min_sample_size
        return min_sample_size*int(chunk_size/min_sample_size)

    def get_batch_size_for_chunk_size(self, chunk_size):
        d = self._params["T"] * self._params["F"]
        assert chunk_size % d == 0
        return chunk_size//d

    def stft(self, waveform, inverse=False):
        N = self._params["frame_length"]
        H = self._params["frame_step"]
        win = hann(N, sym=False)
        fstft = istft if inverse else stft
        win_len_arg = "win_length" if inverse else "n_fft"
        s1 = fstft(waveform[:, 0], hop_length=H, window=win, center=False, **{win_len_arg: N})
        s2 = fstft(waveform[:, 1], hop_length=H, window=win, center=False, **{win_len_arg: N})
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
                out[inst] = self.stft(outputs[inst], inverse=True)
        return out

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
        if self._params["stft_backend"] == "tensorflow":
            sources = self.separate(waveform)
        else:
            sources = self.separate_librosa(waveform, audio_descriptor)
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
