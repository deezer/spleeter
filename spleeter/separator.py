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


from . import SpleeterError
from .audio.adapter import get_default_audio_adapter
from .audio.convertor import to_stereo
from .utils.configuration import load_configuration
from .utils.estimator import create_estimator, to_predictor, get_input_dict_placeholders, get_default_model_dir
from .model import EstimatorSpecBuilder


__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


logger = logging.getLogger("spleeter")


class Separator(object):
    """ A wrapper class for performing separation. """

    def __init__(self, params_descriptor, MWF=False, multiprocess=True):
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

    def separate_chunked(self, waveform, sample_rate, chunk_max_duration):
        chunk_size = self.get_valid_chunk_size(sample_rate, chunk_max_duration)
        print(f"chunk size is {chunk_size}")
        batch_size = self.get_batch_size_for_chunk_size(chunk_size)
        print(f"batch size {batch_size}")
        T, F = self._params["T"], self._params["F"]
        out = {}
        n_batches = (waveform.shape[0]+batch_size*T*F-1)//(batch_size*T*F)
        print(f"{n_batches} to compute")
        features = get_input_dict_placeholders(self._params)
        spectrogram_input_t = tf.placeholder(tf.float32, shape=(None, T, F, 2), name="spectrogram_input")
        istft_input_t = tf.placeholder(tf.complex64, shape=(None, F, 2), name="istft_input")
        start_t = tf.placeholder(tf.int32, shape=(), name="start")
        end_t = tf.placeholder(tf.int32, shape=(), name="end")
        builder = EstimatorSpecBuilder(features, self._params)
        latest_checkpoint = tf.train.latest_checkpoint(get_default_model_dir(self._params['model_dir']))

        # TODO: fix the logic, build sometimes return, sometimes set attribute
        builder._build_stft_feature()
        stft_t = builder.get_stft_feature()
        output_dict_t = builder._build_output_dict(input_tensor=spectrogram_input_t)
        masked_stft_t = builder._build_masked_stft(builder._build_masks(output_dict_t),
                                                   input_stft=stft_t[start_t:end_t, :, :])
        output_waveform_t = builder._inverse_stft(istft_input_t)
        waveform_t = features["waveform"]
        masked_stfts = {}
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("restoring weights {}".format(time()))
            saver.restore(sess, latest_checkpoint)
            print("computing spectrogram {}".format(time()))
            spectrogram, stft = sess.run([builder.get_spectrogram_feature(), stft_t], feed_dict={waveform_t: waveform})
            print(spectrogram.shape)
            print(stft.shape)
            for i in range(n_batches):
                print("computing batch {} {}".format(i, time()))
                start = i*batch_size
                end = (i+1)*batch_size
                tmp = sess.run(masked_stft_t,
                               feed_dict={spectrogram_input_t: spectrogram[start:end, ...],
                                          start_t: start*T, end_t: end*T, stft_t: stft})
                for instrument, masked_stft in tmp.items():
                    masked_stfts.setdefault(instrument, []).append(masked_stft)

            print("inverting spectrogram {}".format(time()))
            for instrument, masked_stft in masked_stfts.items():
                out[instrument] = sess.run(output_waveform_t, {istft_input_t: np.concatenate(masked_stft, axis=0)})
        print("done separating {}".format(time()))
        return out

    def separate_to_file(
            self, audio_descriptor, destination,
            audio_adapter=get_default_audio_adapter(), chunk_duration=-1,
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
        print("loading audio {}".format(time()))
        waveform, sample_rate = audio_adapter.load(
            audio_descriptor,
            offset=offset,
            duration=duration,
            sample_rate=self._sample_rate)
        print("done loading audio {}".format(time()))
        sources = self.separate_chunked(waveform, sample_rate, chunk_duration)
        print("saving to file {}".format(time()))
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
