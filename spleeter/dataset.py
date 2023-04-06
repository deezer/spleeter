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

import os
import time
from os.path import exists
from os.path import sep as SEPARATOR
from typing import Any, Dict, List, Optional, Tuple

# pyright: reportMissingImports=false
# pylint: disable=import-error
import tensorflow as tf  # type: ignore

from .audio.adapter import AudioAdapter
from .audio.convertor import db_uint_spectrogram_to_gain, spectrogram_to_db_uint
from .audio.spectrogram import (
    compute_spectrogram_tf,
    random_pitch_shift,
    random_time_stretch,
)
from .utils.logging import logger
from .utils.tensor import (
    check_tensor_shape,
    dataset_from_csv,
    set_tensor_shape,
    sync_apply,
)

# pylint: enable=import-error

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

# Default audio parameters to use.
DEFAULT_AUDIO_PARAMS: Dict = {
    "instrument_list": ("vocals", "accompaniment"),
    "mix_name": "mix",
    "sample_rate": 44100,
    "frame_length": 4096,
    "frame_step": 1024,
    "T": 512,
    "F": 1024,
}


def get_training_dataset(
    audio_params: Dict, audio_adapter: AudioAdapter, audio_path: str
) -> Any:
    """
    Builds training dataset.

    Parameters:
        audio_params (Dict):
            Audio parameters.
        audio_adapter (AudioAdapter):
            Adapter to load audio from.
        audio_path (str):
            Path of directory containing audio.

    Returns:
        Any:
            Built dataset.
    """
    builder = DatasetBuilder(
        audio_params,
        audio_adapter,
        audio_path,
        chunk_duration=audio_params.get("chunk_duration", 20.0),
        random_seed=audio_params.get("random_seed", 0),
    )
    return builder.build(
        str(audio_params.get("train_csv")),
        cache_directory=audio_params.get("training_cache"),
        batch_size=audio_params.get("batch_size", 8),
        n_chunks_per_song=audio_params.get("n_chunks_per_song", 2),
        random_data_augmentation=False,
        convert_to_uint=True,
        wait_for_cache=False,
    )


def get_validation_dataset(
    audio_params: Dict, audio_adapter: AudioAdapter, audio_path: str
) -> Any:
    """
    Builds validation dataset.

    Parameters:
        audio_params (Dict):
            Audio parameters.
        audio_adapter (AudioAdapter):
            Adapter to load audio from.
        audio_path (str):
            Path of directory containing audio.

    Returns:
        Any:
            Built dataset.
    """
    builder = DatasetBuilder(
        audio_params, audio_adapter, audio_path, chunk_duration=12.0
    )
    return builder.build(
        str(audio_params.get("validation_csv")),
        batch_size=audio_params.get("batch_size", 8),
        cache_directory=audio_params.get("validation_cache"),
        convert_to_uint=True,
        infinite_generator=False,
        n_chunks_per_song=1,
        # should not perform data augmentation for eval:
        random_data_augmentation=False,
        random_time_crop=False,
        shuffle=False,
    )


class InstrumentDatasetBuilder(object):
    """Instrument based filter and mapper provider."""

    def __init__(self, parent: Any, instrument: Any) -> None:
        """
        Default constructor.

        Parameters:
            parent (Any):
                Parent dataset builder.
            instrument (Any):
                Target instrument.
        """
        self._parent = parent
        self._instrument = instrument
        self._spectrogram_key = f"{instrument}_spectrogram"
        self._min_spectrogram_key = f"min_{instrument}_spectrogram"
        self._max_spectrogram_key = f"max_{instrument}_spectrogram"

    def load_waveform(self, sample: Dict) -> Dict:
        """Load waveform for given sample."""
        return dict(
            sample,
            **self._parent._audio_adapter.load_waveform(
                sample[f"{self._instrument}_path"],
                offset=sample["start"],
                duration=self._parent._chunk_duration,
                sample_rate=self._parent._sample_rate,
                waveform_name="waveform",
            ),
        )

    def compute_spectrogram(self, sample: Dict) -> Dict:
        """Compute spectrogram of the given sample."""
        return dict(
            sample,
            **{
                self._spectrogram_key: compute_spectrogram_tf(
                    sample["waveform"],
                    frame_length=self._parent._frame_length,
                    frame_step=self._parent._frame_step,
                    spec_exponent=1.0,
                    window_exponent=1.0,
                )
            },
        )

    def filter_frequencies(self, sample: Dict) -> Dict:
        return dict(
            sample,
            **{
                self._spectrogram_key: sample[self._spectrogram_key][
                    :, : self._parent._F, :
                ]
            },
        )

    def convert_to_uint(self, sample: Dict) -> Dict:
        """Convert given sample from float to unit."""
        return dict(
            sample,
            **spectrogram_to_db_uint(
                sample[self._spectrogram_key],
                tensor_key=self._spectrogram_key,
                min_key=self._min_spectrogram_key,
                max_key=self._max_spectrogram_key,
            ),
        )

    def filter_infinity(self, sample: Dict) -> tf.Tensor:
        """Filter infinity sample."""
        return tf.logical_not(tf.math.is_inf(sample[self._min_spectrogram_key]))

    def convert_to_float32(self, sample: Dict) -> Dict:
        """Convert given sample from unit to float."""
        return dict(
            sample,
            **{
                self._spectrogram_key: db_uint_spectrogram_to_gain(
                    sample[self._spectrogram_key],
                    sample[self._min_spectrogram_key],
                    sample[self._max_spectrogram_key],
                )
            },
        )

    def time_crop(self, sample: Dict) -> Dict:
        def start(sample):
            """mid_segment_start"""
            return tf.cast(
                tf.maximum(
                    tf.shape(sample[self._spectrogram_key])[0] / 2
                    - self._parent._T / 2,
                    0,
                ),
                tf.int32,
            )

        return dict(
            sample,
            **{
                self._spectrogram_key: sample[self._spectrogram_key][
                    start(sample) : start(sample) + self._parent._T, :, :
                ]
            },
        )

    def filter_shape(self, sample: Dict) -> bool:
        """Filter badly shaped sample."""
        return check_tensor_shape(
            sample[self._spectrogram_key],
            (self._parent._T, self._parent._F, self._parent._n_channels),
        )

    def reshape_spectrogram(self, sample: Dict) -> Dict:
        """Reshape given sample."""
        return dict(
            sample,
            **{
                self._spectrogram_key: set_tensor_shape(
                    sample[self._spectrogram_key],
                    (self._parent._T, self._parent._F, self._parent._n_channels),
                )
            },
        )


class DatasetBuilder(object):
    MARGIN: float = 0.5
    """Margin at beginning and end of songs in seconds."""

    WAIT_PERIOD: int = 60
    """Wait period for cache (in seconds)."""

    def __init__(
        self,
        audio_params: Dict,
        audio_adapter: AudioAdapter,
        audio_path: str,
        random_seed: int = 0,
        chunk_duration: float = 20.0,
    ) -> None:
        """
        Default constructor.
        """
        # Length of segment in frames (if fs=22050 and
        # frame_step=512, then T=512 corresponds to 11.89s)
        self._T = audio_params["T"]
        # Number of frequency bins to be used (should
        # be less than frame_length/2 + 1)
        self._F = audio_params["F"]
        self._sample_rate = audio_params["sample_rate"]
        self._frame_length = audio_params["frame_length"]
        self._frame_step = audio_params["frame_step"]
        self._mix_name = audio_params["mix_name"]
        self._n_channels = audio_params["n_channels"]
        self._instruments = [self._mix_name] + audio_params["instrument_list"]
        self._instrument_builders: Optional[List] = None
        self._chunk_duration = chunk_duration
        self._audio_adapter = audio_adapter
        self._audio_params = audio_params
        self._audio_path = audio_path
        self._random_seed = random_seed

        self.check_parameters_compatibility()

    def check_parameters_compatibility(self):
        if self._frame_length / 2 + 1 < self._F:
            raise ValueError(
                "F is too large and must be set to at most frame_length/2+1. "
                "Decrease F or increase frame_length to fix."
            )

        if (
            self._chunk_duration * self._sample_rate - self._frame_length
        ) / self._frame_step < self._T:
            raise ValueError(
                "T is too large considering STFT parameters and chunk duratoin. "
                "Make sure spectrogram time dimension of chunks is larger than T "
                "(for instance reducing T or frame_step or increasing chunk duration)."
            )

    def expand_path(self, sample: Dict) -> Dict:
        """Expands audio paths for the given sample."""
        return dict(
            sample,
            **{
                f"{instrument}_path": tf.strings.join(
                    (self._audio_path, sample[f"{instrument}_path"]), SEPARATOR
                )
                for instrument in self._instruments
            },
        )

    def filter_error(self, sample: Dict) -> tf.Tensor:
        """Filter errored sample."""
        return tf.logical_not(sample["waveform_error"])

    def filter_waveform(self, sample: Dict) -> Dict:
        """Filter waveform from sample."""
        return {k: v for k, v in sample.items() if not k == "waveform"}

    def harmonize_spectrogram(self, sample: Dict) -> Dict:
        """Ensure same size for vocals and mix spectrograms."""

        def _reduce(sample):
            return tf.reduce_min(
                [
                    tf.shape(sample[f"{instrument}_spectrogram"])[0]
                    for instrument in self._instruments
                ]
            )

        return dict(
            sample,
            **{
                f"{instrument}_spectrogram": sample[f"{instrument}_spectrogram"][
                    : _reduce(sample), :, :
                ]
                for instrument in self._instruments
            },
        )

    def filter_short_segments(self, sample: Dict) -> tf.Tensor:
        """Filter out too short segment."""
        return tf.reduce_any(
            [
                tf.shape(sample[f"{instrument}_spectrogram"])[0] >= self._T
                for instrument in self._instruments
            ]
        )

    def random_time_crop(self, sample: Dict) -> Dict:
        """Random time crop of 11.88s."""
        return dict(
            sample,
            **sync_apply(
                {
                    f"{instrument}_spectrogram": sample[f"{instrument}_spectrogram"]
                    for instrument in self._instruments
                },
                lambda x: tf.image.random_crop(
                    x,
                    (self._T, len(self._instruments) * self._F, self._n_channels),
                    seed=self._random_seed,
                ),
            ),
        )

    def random_time_stretch(self, sample: Dict) -> Dict:
        """Randomly time stretch the given sample."""
        return dict(
            sample,
            **sync_apply(
                {
                    f"{instrument}_spectrogram": sample[f"{instrument}_spectrogram"]
                    for instrument in self._instruments
                },
                lambda x: random_time_stretch(x, factor_min=0.9, factor_max=1.1),
            ),
        )

    def random_pitch_shift(self, sample: Dict) -> Dict:
        """Randomly pitch shift the given sample."""
        return dict(
            sample,
            **sync_apply(
                {
                    f"{instrument}_spectrogram": sample[f"{instrument}_spectrogram"]
                    for instrument in self._instruments
                },
                lambda x: random_pitch_shift(x, shift_min=-1.0, shift_max=1.0),
                concat_axis=0,
            ),
        )

    def map_features(self, sample: Dict) -> Tuple[Dict, Dict]:
        """Select features and annotation of the given sample."""
        input_ = {
            f"{self._mix_name}_spectrogram": sample[f"{self._mix_name}_spectrogram"]
        }
        output = {
            f"{instrument}_spectrogram": sample[f"{instrument}_spectrogram"]
            for instrument in self._audio_params["instrument_list"]
        }
        return (input_, output)

    def compute_segments(self, dataset: Any, n_chunks_per_song: int) -> Any:
        """
        Computes segments for each song of the dataset.

        Parameters:
            dataset (Any):
                Dataset to compute segments for.
            n_chunks_per_song (int):
                Number of segment per song to compute.

        Returns:
            Any:
                Segmented dataset.
        """
        if n_chunks_per_song <= 0:
            raise ValueError("n_chunks_per_song must be positif")
        datasets = []
        for k in range(n_chunks_per_song):
            if n_chunks_per_song > 1:
                datasets.append(
                    dataset.map(
                        lambda sample: dict(
                            sample,
                            start=tf.maximum(
                                k
                                * (
                                    sample["duration"]
                                    - self._chunk_duration
                                    - 2 * self.MARGIN
                                )
                                / (n_chunks_per_song - 1)
                                + self.MARGIN,
                                0,
                            ),
                        )
                    )
                )
            elif n_chunks_per_song == 1:  # Take central segment.
                datasets.append(
                    dataset.map(
                        lambda sample: dict(
                            sample,
                            start=tf.maximum(
                                sample["duration"] / 2 - self._chunk_duration / 2, 0
                            ),
                        )
                    )
                )
        dataset = datasets[-1]
        for d in datasets[:-1]:
            dataset = dataset.concatenate(d)
        return dataset

    @property
    def instruments(self) -> Any:
        """
        Instrument dataset builder generator.

        Yields:
            Any:
                InstrumentBuilder instance.
        """
        if self._instrument_builders is None:
            self._instrument_builders = []
            for instrument in self._instruments:
                self._instrument_builders.append(
                    InstrumentDatasetBuilder(self, instrument)
                )
        for builder in self._instrument_builders:
            yield builder

    def cache(self, dataset: Any, cache: Optional[str], wait: bool) -> Any:
        """
        Cache the given dataset if cache is enabled. Eventually waits for
        cache to be available (useful if another process is already
        computing cache) if provided wait flag is `True`.

        Parameters:
            dataset (Any):
                Dataset to be cached if cache is required.
            cache (str):
                Path of cache directory to be used, None if no cache.
            wait (bool):
                If caching is enabled, True is cache should be waited.

        Returns:
            Any:
                Cached dataset if needed, original dataset otherwise.
        """
        if cache is not None:
            if wait:
                while not exists(f"{cache}.index"):
                    logger.info(f"Cache not available, wait {self.WAIT_PERIOD}")
                    time.sleep(self.WAIT_PERIOD)
            cache_path = os.path.split(cache)[0]
            os.makedirs(cache_path, exist_ok=True)
            return dataset.cache(cache)
        return dataset

    def build(
        self,
        csv_path: str,
        batch_size: int = 8,
        shuffle: bool = True,
        convert_to_uint: bool = True,
        random_data_augmentation: bool = False,
        random_time_crop: bool = True,
        infinite_generator: bool = True,
        cache_directory: Optional[str] = None,
        wait_for_cache: bool = False,
        num_parallel_calls: int = 4,
        n_chunks_per_song: int = 2,
    ) -> Any:
        dataset = dataset_from_csv(csv_path)
        dataset = self.compute_segments(dataset, n_chunks_per_song)
        # Shuffle data
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=200000,
                seed=self._random_seed,
                # useless since it is cached :
                reshuffle_each_iteration=True,
            )
        # Expand audio path.
        dataset = dataset.map(self.expand_path)
        # Load waveform, compute spectrogram, and filtering error,
        # K bins frequencies, and waveform.
        N = num_parallel_calls
        for instrument in self.instruments:
            dataset = (
                dataset.map(instrument.load_waveform, num_parallel_calls=N)
                .filter(self.filter_error)
                .map(instrument.compute_spectrogram, num_parallel_calls=N)
                .map(instrument.filter_frequencies)
            )
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
                buffer_size=256, seed=self._random_seed, reshuffle_each_iteration=True
            )
        # Convert back to float32
        if convert_to_uint:
            for instrument in self.instruments:
                dataset = dataset.map(
                    instrument.convert_to_float32, num_parallel_calls=N
                )
        M = 8  # Parallel call post caching.
        # Must be applied with the same factor on mix and vocals.
        if random_data_augmentation:
            dataset = dataset.map(self.random_time_stretch, num_parallel_calls=M).map(
                self.random_pitch_shift, num_parallel_calls=M
            )
        # Filter by shape (remove badly shaped tensors).
        for instrument in self.instruments:
            dataset = dataset.filter(instrument.filter_shape).map(
                instrument.reshape_spectrogram
            )
        # Select features and annotation.
        dataset = dataset.map(self.map_features)
        # Make batch (done after selection to avoid
        # error due to unprocessed instrument spectrogram batching).
        dataset = dataset.batch(batch_size)
        return dataset
