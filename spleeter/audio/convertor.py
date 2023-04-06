#!/usr/bin/env python
# coding: utf8

""" This module provides audio data convertion functions. """

# pyright: reportMissingImports=false
# pylint: disable=import-error
import numpy as np
import tensorflow as tf  # type: ignore

from ..utils.tensor import from_float32_to_uint8, from_uint8_to_float32

# pylint: enable=import-error

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


def to_n_channels(waveform: tf.Tensor, n_channels: int) -> tf.Tensor:
    """
    Convert a waveform to n_channels by removing or duplicating channels if
    needed (in tensorflow).

    Parameters:
        waveform (tf.Tensor):
            Waveform to transform.
        n_channels (int):
            Number of channel to reshape waveform in.

    Returns:
        tf.Tensor:
            Reshaped waveform.
    """
    return tf.cond(
        tf.shape(waveform)[1] >= n_channels,
        true_fn=lambda: waveform[:, :n_channels],
        false_fn=lambda: tf.tile(waveform, [1, n_channels])[:, :n_channels],
    )


def to_stereo(waveform: np.ndarray) -> np.ndarray:
    """
    Convert a waveform to stereo by duplicating if mono, or truncating
    if too many channels.

    Parameters:
        waveform (np.ndarray):
            a `(N, d)` numpy array.

    Returns:
        np.ndarray:
            A stereo waveform as a `(N, 1)` numpy array.
    """
    if waveform.shape[1] == 1:
        return np.repeat(waveform, 2, axis=-1)
    if waveform.shape[1] > 2:
        return waveform[:, :2]
    return waveform


def gain_to_db(tensor: tf.Tensor, espilon: float = 10e-10) -> tf.Tensor:
    """
    Convert from gain to decibel in tensorflow.

    Parameters:
        tensor (tf.Tensor):
            Tensor to convert
        epsilon (float):
            Operation constant.

    Returns:
        tf.Tensor:
            Converted tensor.
    """
    return 20.0 / np.log(10) * tf.math.log(tf.maximum(tensor, espilon))


def db_to_gain(tensor: tf.Tensor) -> tf.Tensor:
    """
    Convert from decibel to gain in tensorflow.

    Parameters:
        tensor (tf.Tensor):
            Tensor to convert

    Returns:
        tf.Tensor:
            Converted tensor.
    """
    return tf.pow(10.0, (tensor / 20.0))


def spectrogram_to_db_uint(
    spectrogram: tf.Tensor, db_range: float = 100.0, **kwargs
) -> tf.Tensor:
    """
    Encodes given spectrogram into uint8 using decibel scale.

    Parameters:
        spectrogram (tf.Tensor):
            Spectrogram to be encoded as TF float tensor.
        db_range (float):
            Range in decibel for encoding.

    Returns:
        tf.Tensor:
            Encoded decibel spectrogram as `uint8` tensor.
    """
    db_spectrogram: tf.Tensor = gain_to_db(spectrogram)
    max_db_spectrogram: tf.Tensor = tf.reduce_max(db_spectrogram)
    int_db_spectrogram: tf.Tensor = tf.maximum(
        db_spectrogram, max_db_spectrogram - db_range
    )
    return from_float32_to_uint8(int_db_spectrogram, **kwargs)


def db_uint_spectrogram_to_gain(
    db_uint_spectrogram: tf.Tensor, min_db: tf.Tensor, max_db: tf.Tensor
) -> tf.Tensor:
    """
    Decode spectrogram from uint8 decibel scale.

    Paramters:
        db_uint_spectrogram (tf.Tensor):
            Decibel spectrogram to decode.
        min_db (tf.Tensor):
            Lower bound limit for decoding.
        max_db (tf.Tensor):
            Upper bound limit for decoding.

    Returns:
        tf.Tensor:
            Decoded spectrogram as `float32` tensor.
    """
    db_spectrogram: tf.Tensor = from_uint8_to_float32(
        db_uint_spectrogram, min_db, max_db
    )
    return db_to_gain(db_spectrogram)
