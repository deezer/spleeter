#!/usr/bin/env python
# coding: utf8

""" This package provide model functions. """

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


def apply(function, input_tensor, instruments, params={}):
    """ Apply given function to the input tensor.

    :param function: Function to be applied to tensor.
    :param input_tensor: Tensor to apply blstm to.
    :param instruments: Iterable that provides a collection of instruments.
    :param params: (Optional) dict of BLSTM parameters.
    :returns: Created output tensor dict.
    """
    output_dict = {}
    for instrument in instruments:
        out_name = f'{instrument}_spectrogram'
        output_dict[out_name] = function(
            input_tensor,
            output_name=out_name,
            params=params)
    return output_dict
