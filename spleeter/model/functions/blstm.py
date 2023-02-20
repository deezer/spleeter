#!/usr/bin/env python
# coding: utf8

"""
This system (UHL1) uses a bi-directional LSTM network as described in :

`S. Uhlich, M. Porcu, F. Giron, M. Enenkl, T. Kemp, N. Takahashi and
Y. Mitsufuji.

"Improving music source separation based on deep neural networks through
data augmentation and network blending", Proc. ICASSP, 2017.`

It has three BLSTM layers, each having 500 cells.  For each instrument,
a network is trained which predicts the target instrument amplitude from
the mixture amplitude in the STFT domain (frame size: 4096, hop size:
1024). The raw output of each network is then combined by a multichannel
Wiener filter. The network is trained on musdb where we split train into
train_train and train_valid with 86 and 14 songs, respectively. The
validation set is used to perform early stopping and hyperparameter
selection (LSTM layer dropout rate, regularization strength).
"""

from typing import Dict, Optional

# pyright: reportMissingImports=false
# pylint: disable=import-error
import tensorflow as tf  # type: ignore
from tensorflow.compat.v1.keras.initializers import he_uniform  # type: ignore
from tensorflow.compat.v1.keras.layers import CuDNNLSTM  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Bidirectional,
    Dense,
    Flatten,
    Reshape,
    TimeDistributed,
)

from . import apply

# pylint: enable=import-error

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


def apply_blstm(
    input_tensor: tf.Tensor, output_name: str = "output", params: Optional[Dict] = None
) -> tf.Tensor:
    """
    Apply BLSTM to the given input_tensor.

    Parameters:
        input_tensor (tf.Tensor):
            Input of the model.
        output_name (str):
            (Optional) name of the output, default to 'output'.
        params (Optional[Dict]):
            (Optional) dict of BLSTM parameters.

    Returns:
        tf.Tensor:
            Output tensor.
    """
    if params is None:
        params = {}
    units: int = params.get("lstm_units", 250)
    kernel_initializer = he_uniform(seed=50)
    flatten_input = TimeDistributed(Flatten())((input_tensor))

    def create_bidirectional():
        return Bidirectional(
            CuDNNLSTM(
                units, kernel_initializer=kernel_initializer, return_sequences=True
            )
        )

    l1 = create_bidirectional()((flatten_input))
    l2 = create_bidirectional()((l1))
    l3 = create_bidirectional()((l2))
    dense = TimeDistributed(
        Dense(
            int(flatten_input.shape[2]),
            activation="relu",
            kernel_initializer=kernel_initializer,
        )
    )((l3))
    output: tf.Tensor = TimeDistributed(
        Reshape(input_tensor.shape[2:]), name=output_name
    )(dense)
    return output


def blstm(
    input_tensor: tf.Tensor, output_name: str = "output", params: Optional[Dict] = None
) -> tf.Tensor:
    """Model function applier."""
    return apply(apply_blstm, input_tensor, output_name, params)
