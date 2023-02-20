#!/usr/bin/env python
# coding: utf8

""" This package provide model functions. """

from typing import Callable, Dict, Iterable, Optional

# pyright: reportMissingImports=false
# pylint: disable=import-error
import tensorflow as tf  # type: ignore

# pylint: enable=import-error

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


def apply(
    function: Callable,
    input_tensor: tf.Tensor,
    instruments: Iterable[str],
    params: Optional[Dict] = None,
) -> Dict:
    """
    Apply given function to the input tensor.

    Parameters:
        function (Callable):
            Function to be applied to tensor.
        input_tensor (tf.Tensor):
            Tensor to apply blstm to.
        instruments (Iterable[str]):
            Iterable that provides a collection of instruments.
        params (Optional[Dict]):
            (Optional) dict of BLSTM parameters.

    Returns:
        Dict:
            Created output tensor dict.
    """
    output_dict: Dict = {}
    for instrument in instruments:
        out_name = f"{instrument}_spectrogram"
        output_dict[out_name] = function(
            input_tensor, output_name=out_name, params=params or {}
        )
    return output_dict
