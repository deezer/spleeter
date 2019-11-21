#!/usr/bin/env python
# coding: utf8

"""
    Module for building data preprocessing pipeline using the tensorflow
    data API. Data preprocessing such as audio loading, spectrogram
    computation, cropping, feature caching or data augmentation is done
    using a tensorflow dataset object that output a tuple (input_, output)
    where:

    -   input is a dictionary with a single key that contains the (batched)
        mix spectrogram of audio samples
    -   output is a dictionary of spectrogram of the isolated tracks
        (ground truth)
"""

import time
import os
from os.path import exists, join, sep as SEPARATOR

# pylint: disable=import-error
import pandas as pd
import numpy as np
import tensorflow as tf
# pylint: enable=import-error

from .audio.convertor import (
    db_uint_spectrogram_to_gain,
    spectrogram_to_db_uint)
from .audio.spectrogram import (
    compute_spectrogram_tf,
    random_pitch_shift,
    random_time_stretch)
from .utils.logging import get_logger
from .utils.tensor import (
    check_tensor_shape,
    dataset_from_csv,
    set_tensor_shape,
    sync_apply)

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

# Default audio parameters to use.
DEFAULT_AUDIO_PARAMS = {
    'instrument_list': ('vocals', 'accompaniment'),
    'mix_name': 'mix',
    'sample_rate': 44100,
    'frame_length': 4096,
    'frame_step': 1024,
    'T': 512,
    'F': 1024
}


def get_training_dataset(audio_params, audio_adapter, audio_path):
    """ Builds training dataset.

    :param audio_params: Audio parameters.
    :param audio_adapter: Adapter to load audio from.
    :param audio_path: Path of directory containing audio.
    :returns: Built dataset.
    """
    builder = DatasetBuilder(
        audio_params,
        audio_adapter,
        audio_path,
        chunk_duration=audio_params.get('chunk_duration', 20.0),
        random_seed=audio_params.get('random_seed', 0))
    return builder.build(
        audio_params.get('train_csv'),
        cache_directory=audio_params.get('training_cache'),
        batch_size=audio_params.get('batch_size'),
        n_chunks_per_song=audio_params.get('n_chunks_per_song', 2),
        random_data_augmentation=False,
        convert_to_uint=True,
        wait_for_cache=False)


def get_validation_dataset(audio_params, audio_adapter, audio_path):
    """ Builds validation dataset.

    :param audio_params: Audio parameters.
    :param audio_adapter: Adapter to load audio from.
    :param audio_path: Path of directory containing audio.
    :returns: Built dataset.
    """
    builder = DatasetBuilder(
        audio_params,
        audio_adapter,
        audio_path,
        chunk_duration=12.0)
    return builder.build(
        audio_params.get('validation_csv'),
        batch_size=audio_params.get('batch_size'),
        cache_directory=audio_params.get('validation_cache'),
        convert_to_uint=True,
        infinite_generator=False,
        n_chunks_per_song=1,
        # should not perform data augmentation for eval:
        random_data_augmentation=False,
        random_time_crop=False,
        shuffle=False,
    )


class InstrumentDatasetBuilder(object):
    """ Instrument based filter and mapper provider. """

    def __init__(self, parent, instrument):
        """ Default constructor.

        :param parent: Parent dataset builder.
        :param instrument: Target instrument.
        """
        self._parent = parent
        self._instrument = instrument
        self._spectrogram_key = f'{instrument}_spectrogram'
        self._min_spectrogram_key = f'min_{instrument}_spectrogram'
        self._max_spectrogram_key = f'max_{instrument}_spectrogram'

    def load_waveform(self, sample):
        """ Load waveform for given sample. """
        return dict(sample, **self._parent._audio_adapter.load_tf_waveform(
            sample[f'{self._instrument}_path'],
            offset=sample['start'],
            duration=self._parent._chunk_duration,
            sample_rate=self._parent._sample_rate,
            waveform_name='waveform'))

    def compute_spectrogram(self, sample):
        """ Compute spectrogram of the given sample. """
        return dict(sample, **{
            self._spectrogram_key: compute_spectrogram_tf(
                sample['waveform'],
                frame_length=self._parent._frame_length,
                frame_step=self._parent._frame_step,
                spec_exponent=1.,
                window_exponent=1.)})

    def filter_frequencies(self, sample):
        """ """
        return dict(sample, **{
            self._spectrogram_key:
                sample[self._spectrogram_key][:, :self._parent._F, :]})

    def convert_to_uint(self, sample):
        """ Convert given sample from float to unit. """
        return dict(sample, **spectrogram_to_db_uint(
            sample[self._spectrogram_key],
            tensor_key=self._spectrogram_key,
            min_key=self._min_spectrogram_key,
            max_key=self._max_spectrogram_key))

    def filter_infinity(self, sample):
        """ Filter infinity sample. """
        return tf.logical_not(
            tf.math.is_inf(
                sample[self._min_spectrogram_key]))

    def convert_to_float32(self, sample):
        """ Convert given sample from unit to float. """
        return dict(sample, **{
            self._spectrogram_key: db_uint_spectrogram_to_gain(
                sample[self._spectrogram_key],
                sample[self._min_spectrogram_key],
                sample[self._max_spectrogram_key])})

    def time_crop(self, sample):
        """ """
        def start(sample):
            """ mid_segment_start """
            return tf.cast(
                tf.maximum(
                    tf.shape(sample[self._spectrogram_key])[0]
                    / 2 - self._parent._T / 2, 0),
                tf.int32)
        return dict(sample, **{
            self._spectrogram_key: sample[self._spectrogram_key][
                start(sample):start(sample) + self._parent._T, :, :]})

    def filter_shape(self, sample):
        """ Filter badly shaped sample. """
        return check_tensor_shape(
            sample[self._spectrogram_key], (
                self._parent._T, self._parent._F, 2))

    def reshape_spectrogram(self, sample):
        """ """
        return dict(sample, **{
            self._spectrogram_key: set_tensor_shape(
                sample[self._spectrogram_key],
                (self._parent._T, self._parent._F, 2))})


class DatasetBuilder(object):
    """
    """

    # Margin at beginning and end of songs in seconds.
    MARGIN = 0.5

    # Wait period for cache (in seconds).
    WAIT_PERIOD = 60

    def __init__(
            self,
            audio_params, audio_adapter, audio_path,
            random_seed=0, chunk_duration=20.0):
        """ Default constructor.

        NOTE: Probably need for AudioAdapter.

        :param audio_params: Audio parameters to use.
        :param audio_adapter: Audio adapter to use.
        :param audio_path:
        :param random_seed:
        :param chunk_duration:
        """
        # Length of segment in frames (if fs=22050 and
        # frame_step=512, then T=512 corresponds to 11.89s)
        self._T = audio_params['T']
        # Number of frequency bins to be used (should
        # be less than frame_length/2 + 1)
        self._F = audio_params['F']
        self._sample_rate = audio_params['sample_rate']
        self._frame_length = audio_params['frame_length']
        self._frame_step = audio_params['frame_step']
        self._mix_name = audio_params['mix_name']
        self._instruments = [self._mix_name] + audio_params['instrument_list']
        self._instrument_builders = None
        self._chunk_duration = chunk_duration
        self._audio_adapter = audio_adapter
        self._audio_params = audio_params
        self._audio_path = audio_path
        self._random_seed = random_seed

    def expand_path(self, sample):
        """ Expands audio paths for the given sample. """
        return dict(sample, **{f'{instrument}_path': tf.string_join(
            (self._audio_path, sample[f'{instrument}_path']), SEPARATOR)
            for instrument in self._instruments})

    def filter_error(self, sample):
        """ Filter errored sample. """
        return tf.logical_not(sample['waveform_error'])

    def filter_waveform(self, sample):
        """ Filter waveform from sample. """
        return {k: v for k, v in sample.items() if not k == 'waveform'}

    def harmonize_spectrogram(self, sample):
        """ Ensure same size for vocals and mix spectrograms. """
        def _reduce(sample):
            return tf.reduce_min([
                tf.shape(sample[f'{instrument}_spectrogram'])[0]
                for instrument in self._instruments])
        return dict(sample, **{
            f'{instrument}_spectrogram':
                sample[f'{instrument}_spectrogram'][:_reduce(sample), :, :]
            for instrument in self._instruments})

    def filter_short_segments(self, sample):
        """ Filter out too short segment. """
        return tf.reduce_any([
            tf.shape(sample[f'{instrument}_spectrogram'])[0] >= self._T
            for instrument in self._instruments])

    def random_time_crop(self, sample):
        """ Random time crop of 11.88s. """
        return dict(sample, **sync_apply({
            f'{instrument}_spectrogram': sample[f'{instrument}_spectrogram']
            for instrument in self._instruments},
            lambda x: tf.image.random_crop(
                x, (self._T, len(self._instruments) * self._F, 2),
                seed=self._random_seed)))

    def random_time_stretch(self, sample):
        """ Randomly time stretch the given sample. """
        return dict(sample, **sync_apply({
            f'{instrument}_spectrogram':
                sample[f'{instrument}_spectrogram']
            for instrument in self._instruments},
            lambda x: random_time_stretch(
                x, factor_min=0.9, factor_max=1.1)))

    def random_pitch_shift(self, sample):
        """ Randomly pitch shift the given sample. """
        return dict(sample, **sync_apply({
            f'{instrument}_spectrogram':
                sample[f'{instrument}_spectrogram']
            for instrument in self._instruments},
            lambda x: random_pitch_shift(
                x, shift_min=-1.0, shift_max=1.0), concat_axis=0))

    def map_features(self, sample):
        """ Select features and annotation of the given sample. """
        input_ = {
            f'{self._mix_name}_spectrogram':
                sample[f'{self._mix_name}_spectrogram']}
        output = {
            f'{instrument}_spectrogram': sample[f'{instrument}_spectrogram']
            for instrument in self._audio_params['instrument_list']}
        return (input_, output)

    def compute_segments(self, dataset, n_chunks_per_song):
        """ Computes segments for each song of the dataset.

        :param dataset: Dataset to compute segments for.
        :param n_chunks_per_song: Number of segment per song to compute.
        :returns: Segmented dataset.
        """
        if n_chunks_per_song <= 0:
            raise ValueError('n_chunks_per_song must be positif')
        datasets = []
        for k in range(n_chunks_per_song):
            if n_chunks_per_song > 1:
                datasets.append(
                    dataset.map(lambda sample: dict(sample, start=tf.maximum(
                        k * (
                            sample['duration'] - self._chunk_duration - 2
                            * self.MARGIN) / (n_chunks_per_song - 1)
                        + self.MARGIN, 0))))
            elif n_chunks_per_song == 1:  # Take central segment.
                datasets.append(
                    dataset.map(lambda sample: dict(sample, start=tf.maximum(
                        sample['duration'] / 2 - self._chunk_duration / 2,
                        0))))
        dataset = datasets[-1]
        for d in datasets[:-1]:
            dataset = dataset.concatenate(d)
        return dataset

    @property
    def instruments(self):
        """ Instrument dataset builder generator.

        :yield InstrumentBuilder instance.
        """
        if self._instrument_builders is None:
            self._instrument_builders = []
            for instrument in self._instruments:
                self._instrument_builders.append(
                    InstrumentDatasetBuilder(self, instrument))
        for builder in self._instrument_builders:
            yield builder

    def cache(self, dataset, cache, wait):
        """ Cache the given dataset if cache is enabled. Eventually waits for
        cache to be available (useful if another process is already computing
        cache) if provided wait flag is True.

        :param dataset: Dataset to be cached if cache is required.
        :param cache: Path of cache directory to be used, None if no cache.
        :param wait: If caching is enabled, True is cache should be waited.
        :returns: Cached dataset if needed, original dataset otherwise.
        """
        if cache is not None:
            if wait:
                while not exists(f'{cache}.index'):
                    get_logger().info(
                        'Cache not available, wait %s',
                        self.WAIT_PERIOD)
                    time.sleep(self.WAIT_PERIOD)
            cache_path = os.path.split(cache)[0]
            os.makedirs(cache_path, exist_ok=True)
            return dataset.cache(cache)
        return dataset

    def build(
            self, csv_path,
            batch_size=8, shuffle=True, convert_to_uint=True,
            random_data_augmentation=False, random_time_crop=True,
            infinite_generator=True, cache_directory=None,
            wait_for_cache=False, num_parallel_calls=4, n_chunks_per_song=2,):
        """
        TO BE DOCUMENTED.
        """
        dataset = dataset_from_csv(csv_path)
        dataset = self.compute_segments(dataset, n_chunks_per_song)
        # Shuffle data
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=200000,
                seed=self._random_seed,
                # useless since it is cached :
                reshuffle_each_iteration=True)
        # Expand audio path.
        dataset = dataset.map(self.expand_path)
        # Load waveform, compute spectrogram, and filtering error,
        # K bins frequencies, and waveform.
        N = num_parallel_calls
        for instrument in self.instruments:
            dataset = (
                dataset
                .map(instrument.load_waveform, num_parallel_calls=N)
                .filter(self.filter_error)
                .map(instrument.compute_spectrogram, num_parallel_calls=N)
                .map(instrument.filter_frequencies))
        dataset = dataset.map(self.filter_waveform)
        # Convert to uint before caching in order to save space.
        if convert_to_uint:
            for instrument in self.instruments:
                dataset = dataset.map(instrument.convert_to_uint)
        dataset = self.cache(dataset, cache_directory, wait_for_cache)
        # Check for INFINITY (should not happen)
        for instrument in self.instruments:
            dataset = dataset.filter(instrument.filter_infinity)
        # Repeat indefinitly
        if infinite_generator:
            dataset = dataset.repeat(count=-1)
        # Ensure same size for vocals and mix spectrograms.
        # NOTE: could be done before caching ?
        dataset = dataset.map(self.harmonize_spectrogram)
        # Filter out too short segment.
        # NOTE: could be done before caching ?
        dataset = dataset.filter(self.filter_short_segments)
        # Random time crop of 11.88s
        if random_time_crop:
            dataset = dataset.map(self.random_time_crop, num_parallel_calls=N)
        else:
            # frame_duration = 11.88/T
            # take central segment (for validation)
            for instrument in self.instruments:
                dataset = dataset.map(instrument.time_crop)
        # Post cache shuffling. Done where the data are the lightest:
        # after croping but before converting back to float.
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=256, seed=self._random_seed,
                reshuffle_each_iteration=True)
        # Convert back to float32
        if convert_to_uint:
            for instrument in self.instruments:
                dataset = dataset.map(
                    instrument.convert_to_float32, num_parallel_calls=N)
        M = 8  # Parallel call post caching.
        # Must be applied with the same factor on mix and vocals.
        if random_data_augmentation:
            dataset = (
                dataset
                .map(self.random_time_stretch, num_parallel_calls=M)
                .map(self.random_pitch_shift, num_parallel_calls=M))
        # Filter by shape (remove badly shaped tensors).
        for instrument in self.instruments:
            dataset = (
                dataset
                .filter(instrument.filter_shape)
                .map(instrument.reshape_spectrogram))
        # Select features and annotation.
        dataset = dataset.map(self.map_features)
        # Make batch (done after selection to avoid
        # error due to unprocessed instrument spectrogram batching).
        dataset = dataset.batch(batch_size)
        return dataset
