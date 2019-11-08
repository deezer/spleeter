#!/usr/bin/env python
# coding: utf8

""" This module provides audio data convertion functions. """

# pylint: disable=import-error
import numpy as np
import tensorflow as tf
# pylint: enable=import-error

from ..utils.tensor import from_float32_to_uint8, from_uint8_to_float32

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


def to_n_channels(waveform, n_channels):
    """ Convert a waveform to n_channels by removing or
    duplicating channels if needed (in tensorflow).

    :param waveform: Waveform to transform.
    :param n_channels: Number of channel to reshape waveform in.
    :returns: Reshaped waveform.
    """
    return tf.cond(
        tf.shape(waveform)[1] >= n_channels,
        true_fn=lambda: waveform[:, :n_channels],
        false_fn=lambda: tf.tile(waveform, [1, n_channels])[:, :n_channels]
    )


def to_stereo(waveform):
    """ Convert a waveform to stereo by duplicating if mono,
    or truncating if too many channels.

    :param waveform: a (N, d) numpy array.
    :returns: A stereo waveform as a (N, 1) numpy array.
    """
    if waveform.shape[1] == 1:
        return np.repeat(waveform, 2, axis=-1)
    if waveform.shape[1] > 2:
        return waveform[:, :2]
    return waveform


def gain_to_db(tensor, espilon=10e-10):
    """ Convert from gain to decibel in tensorflow.

    :param tensor: Tensor to convert.
    :param epsilon: Operation constant.
    :returns: Converted tensor.
    """
    return 20. / np.log(10) * tf.math.log(tf.maximum(tensor, espilon))


def db_to_gain(tensor):
    """ Convert from decibel to gain in tensorflow.

    :param tensor_db: Tensor to convert.
    :returns: Converted tensor.
    """
    return tf.pow(10., (tensor / 20.))


def spectrogram_to_db_uint(spectrogram, db_range=100., **kwargs):
    """ Encodes given spectrogram into uint8 using decibel scale.

    :param spectrogram: Spectrogram to be encoded as TF float tensor.
    :param db_range: Range in decibel for encoding.
    :returns: Encoded decibel spectrogram as uint8 tensor.
    """
    db_spectrogram = gain_to_db(spectrogram)
    max_db_spectrogram = tf.reduce_max(db_spectrogram)
    db_spectrogram = tf.maximum(db_spectrogram, max_db_spectrogram - db_range)
    return from_float32_to_uint8(db_spectrogram, **kwargs)


def db_uint_spectrogram_to_gain(db_uint_spectrogram, min_db, max_db):
    """ Decode spectrogram from uint8 decibel scale.

    :param db_uint_spectrogram: Decibel pectrogram to decode.
    :param min_db: Lower bound limit for decoding.
    :param max_db: Upper bound limit for decoding.
    :returns: Decoded spectrogram as float2 tensor.
    """
    db_spectrogram = from_uint8_to_float32(db_uint_spectrogram, min_db, max_db)
    return db_to_gain(db_spectrogram)
