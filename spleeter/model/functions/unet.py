#!/usr/bin/env python
# coding: utf8

"""
This module contains building functions for U-net source
separation models in a similar way as in A. Jansson et al. "Singing
voice separation with deep u-net convolutional networks", ISMIR 2017.
Each instrument is modeled by a single U-net convolutional
/ deconvolutional network that take a mix spectrogram as input and the
estimated sound spectrogram as output.
"""

from functools import partial

# pylint: disable=import-error
import tensorflow as tf

from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    ELU,
    LeakyReLU,
    Multiply,
    ReLU,
    Softmax)
from tensorflow.compat.v1 import logging
from tensorflow.compat.v1.keras.initializers import he_uniform
# pylint: enable=import-error

from . import apply

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


def _get_conv_activation_layer(params):
    """

    :param params:
    :returns: Required Activation function.
    """
    conv_activation = params.get('conv_activation')
    if conv_activation == 'ReLU':
        return ReLU()
    elif conv_activation == 'ELU':
        return ELU()
    return LeakyReLU(0.2)


def _get_deconv_activation_layer(params):
    """

    :param params:
    :returns: Required Activation function.
    """
    deconv_activation = params.get('deconv_activation')
    if deconv_activation == 'LeakyReLU':
        return LeakyReLU(0.2)
    elif deconv_activation == 'ELU':
        return ELU()
    return ReLU()


def apply_unet(
        input_tensor,
        output_name='output',
        params={},
        output_mask_logit=False):
    """ Apply a convolutionnal U-net to model a single instrument (one U-net
    is used for each instrument).

    :param input_tensor:
    :param output_name: (Optional) , default to 'output'
    :param params: (Optional) , default to empty dict.
    :param output_mask_logit: (Optional) , default to False.
    """
    logging.info(f'Apply unet for {output_name}')
    conv_n_filters = params.get('conv_n_filters', [16, 32, 64, 128, 256, 512])
    conv_activation_layer = _get_conv_activation_layer(params)
    deconv_activation_layer = _get_deconv_activation_layer(params)
    kernel_initializer = he_uniform(seed=50)
    conv2d_factory = partial(
        Conv2D,
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer)
    # First layer.
    conv1 = conv2d_factory(conv_n_filters[0], (5, 5))(input_tensor)
    batch1 = BatchNormalization(axis=-1)(conv1)
    rel1 = conv_activation_layer(batch1)
    # Second layer.
    conv2 = conv2d_factory(conv_n_filters[1], (5, 5))(rel1)
    batch2 = BatchNormalization(axis=-1)(conv2)
    rel2 = conv_activation_layer(batch2)
    # Third layer.
    conv3 = conv2d_factory(conv_n_filters[2], (5, 5))(rel2)
    batch3 = BatchNormalization(axis=-1)(conv3)
    rel3 = conv_activation_layer(batch3)
    # Fourth layer.
    conv4 = conv2d_factory(conv_n_filters[3], (5, 5))(rel3)
    batch4 = BatchNormalization(axis=-1)(conv4)
    rel4 = conv_activation_layer(batch4)
    # Fifth layer.
    conv5 = conv2d_factory(conv_n_filters[4], (5, 5))(rel4)
    batch5 = BatchNormalization(axis=-1)(conv5)
    rel5 = conv_activation_layer(batch5)
    # Sixth layer
    conv6 = conv2d_factory(conv_n_filters[5], (5, 5))(rel5)
    batch6 = BatchNormalization(axis=-1)(conv6)
    _ = conv_activation_layer(batch6)
    #
    #
    conv2d_transpose_factory = partial(
        Conv2DTranspose,
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer)
    #
    up1 = conv2d_transpose_factory(conv_n_filters[4], (5, 5))((conv6))
    up1 = deconv_activation_layer(up1)
    batch7 = BatchNormalization(axis=-1)(up1)
    drop1 = Dropout(0.5)(batch7)
    merge1 = Concatenate(axis=-1)([conv5, drop1])
    #
    up2 = conv2d_transpose_factory(conv_n_filters[3], (5, 5))((merge1))
    up2 = deconv_activation_layer(up2)
    batch8 = BatchNormalization(axis=-1)(up2)
    drop2 = Dropout(0.5)(batch8)
    merge2 = Concatenate(axis=-1)([conv4, drop2])
    #
    up3 = conv2d_transpose_factory(conv_n_filters[2], (5, 5))((merge2))
    up3 = deconv_activation_layer(up3)
    batch9 = BatchNormalization(axis=-1)(up3)
    drop3 = Dropout(0.5)(batch9)
    merge3 = Concatenate(axis=-1)([conv3, drop3])
    #
    up4 = conv2d_transpose_factory(conv_n_filters[1], (5, 5))((merge3))
    up4 = deconv_activation_layer(up4)
    batch10 = BatchNormalization(axis=-1)(up4)
    merge4 = Concatenate(axis=-1)([conv2, batch10])
    #
    up5 = conv2d_transpose_factory(conv_n_filters[0], (5, 5))((merge4))
    up5 = deconv_activation_layer(up5)
    batch11 = BatchNormalization(axis=-1)(up5)
    merge5 = Concatenate(axis=-1)([conv1, batch11])
    #
    up6 = conv2d_transpose_factory(1, (5, 5), strides=(2, 2))((merge5))
    up6 = deconv_activation_layer(up6)
    batch12 = BatchNormalization(axis=-1)(up6)
    # Last layer to ensure initial shape reconstruction.
    if not output_mask_logit:
        up7 = Conv2D(
            2,
            (4, 4),
            dilation_rate=(2, 2),
            activation='sigmoid',
            padding='same',
            kernel_initializer=kernel_initializer)((batch12))
        output = Multiply(name=output_name)([up7, input_tensor])
        return output
    return Conv2D(
        2,
        (4, 4),
        dilation_rate=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer)((batch12))


def unet(input_tensor, instruments, params={}):
    """ Model function applier. """
    return apply(apply_unet, input_tensor, instruments, params)


def softmax_unet(input_tensor, instruments, params={}):
    """ Apply softmax to multitrack unet in order to have mask suming to one.

    :param input_tensor: Tensor to apply blstm to.
    :param instruments: Iterable that provides a collection of instruments.
    :param params: (Optional) dict of BLSTM parameters.
    :returns: Created output tensor dict.
    """
    logit_mask_list = []
    for instrument in instruments:
        out_name = f'{instrument}_spectrogram'
        logit_mask_list.append(
            apply_unet(
                input_tensor,
                output_name=out_name,
                params=params,
                output_mask_logit=True))
    masks = Softmax(axis=4)(tf.stack(logit_mask_list, axis=4))
    output_dict = {}
    for i, instrument in enumerate(instruments):
        out_name = f'{instrument}_spectrogram'
        output_dict[out_name] = Multiply(name=out_name)([
            masks[..., i],
            input_tensor])
    return output_dict
