#!/usr/bin/env python
# coding: utf8

""" Utility function for tensorflow. """

# pylint: disable=import-error
import tensorflow as tf
import pandas as pd
# pylint: enable=import-error

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


def sync_apply(tensor_dict, func, concat_axis=1):
    """ Return a function that applies synchronously the provided func on the
    provided dictionnary of tensor. This means that func is applied to the
    concatenation of the tensors in tensor_dict. This is useful for performing
    random operation that needs the same drawn value on multiple tensor, such
    as a random time-crop on both input data and label (the same crop should be
    applied to both input data and label, so random crop cannot be applied
    separately on each of them).

    IMPORTANT NOTE: all tensor are assumed to be the same shape.

    Params:
        - tensor_dict: dictionary (key: strings, values: tf.tensor)
        a dictionary of tensor.
        - func: function
        function to be applied to the concatenation of the tensors in
        tensor_dict
        - concat_axis: int
        The axis on which to perform the concatenation.

        Returns:
        processed tensors dictionary with the same name (keys) as input
        tensor_dict.
    """
    if concat_axis not in {0, 1}:
        raise NotImplementedError(
            'Function only implemented for concat_axis equal to 0 or 1')
    tensor_list = list(tensor_dict.values())
    concat_tensor = tf.concat(tensor_list, concat_axis)
    processed_concat_tensor = func(concat_tensor)
    tensor_shape = tf.shape(list(tensor_dict.values())[0])
    D = tensor_shape[concat_axis]
    if concat_axis == 0:
        return {
            name: processed_concat_tensor[index * D:(index + 1) * D, :, :]
            for index, name in enumerate(tensor_dict)
        }
    return {
        name: processed_concat_tensor[:, index * D:(index + 1) * D, :]
        for index, name in enumerate(tensor_dict)
    }


def from_float32_to_uint8(
        tensor,
        tensor_key='tensor',
        min_key='min',
        max_key='max'):
    """

    :param tensor:
    :param tensor_key:
    :param min_key:
    :param max_key:
    :returns:
    """
    tensor_min = tf.reduce_min(tensor)
    tensor_max = tf.reduce_max(tensor)
    return {
        tensor_key: tf.cast(
            (tensor - tensor_min) / (tensor_max - tensor_min + 1e-16)
            * 255.9999, dtype=tf.uint8),
        min_key: tensor_min,
        max_key: tensor_max
    }


def from_uint8_to_float32(tensor, tensor_min, tensor_max):
    """

    :param tensor:
    :param tensor_min:
    :param tensor_max:
    :returns:
    """
    return (
        tf.cast(tensor, tf.float32)
        * (tensor_max - tensor_min)
        / 255.9999 + tensor_min)


def pad_and_partition(tensor, segment_len):
    """ Pad and partition a tensor into segment of len segment_len
    along the first dimension. The tensor is padded with 0 in order
    to ensure that the first dimension is a multiple of segment_len.

    Tensor must be of known fixed rank

    :Example:

    >>> tensor = [[1, 2, 3], [4, 5, 6]]
    >>> segment_len = 2
    >>> pad_and_partition(tensor, segment_len)
    [[[1, 2], [4, 5]], [[3, 0], [6, 0]]]

    :param tensor:
    :param segment_len:
    :returns:
    """
    tensor_size = tf.math.floormod(tf.shape(tensor)[0], segment_len)
    pad_size = tf.math.floormod(segment_len - tensor_size, segment_len)
    padded = tf.pad(
        tensor,
        [[0, pad_size]] + [[0, 0]] * (len(tensor.shape)-1))
    split = (tf.shape(padded)[0] + segment_len - 1) // segment_len
    return tf.reshape(
        padded,
        tf.concat(
            [[split, segment_len], tf.shape(padded)[1:]],
            axis=0))


def pad_and_reshape(instr_spec, frame_length, F):
    """
    :param instr_spec:
    :param frame_length:
    :param F:
    :returns:
    """
    spec_shape = tf.shape(instr_spec)
    extension_row = tf.zeros((spec_shape[0], spec_shape[1], 1, spec_shape[-1]))
    n_extra_row = (frame_length) // 2 + 1 - F
    extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
    extended_spec = tf.concat([instr_spec, extension], axis=2)
    old_shape = tf.shape(extended_spec)
    new_shape = tf.concat([
        [old_shape[0] * old_shape[1]],
        old_shape[2:]],
        axis=0)
    processed_instr_spec = tf.reshape(extended_spec, new_shape)
    return processed_instr_spec


def dataset_from_csv(csv_path, **kwargs):
    """ Load dataset from a CSV file using Pandas. kwargs if any are
    forwarded to the `pandas.read_csv` function.

    :param csv_path: Path of the CSV file to load dataset from.
    :returns: Loaded dataset.
    """
    df = pd.read_csv(csv_path, **kwargs)
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            {key: df[key].values for key in df})
    )
    return dataset


def check_tensor_shape(tensor_tf, target_shape):
    """ Return a Tensorflow boolean graph that indicates whether
    sample[features_key] has the specified target shape. Only check
    not None entries of target_shape.

    :param tensor_tf: Tensor to check shape for.
    :param target_shape: Target shape to compare tensor to.
    :returns: True if shape is valid, False otherwise (as TF boolean).
    """
    result = tf.constant(True)
    for i, target_length in enumerate(target_shape):
        if target_length:
            result = tf.logical_and(
                result,
                tf.equal(tf.constant(target_length), tf.shape(tensor_tf)[i]))
    return result


def set_tensor_shape(tensor, tensor_shape):
    """ Set shape for a tensor (not in place, as opposed to tf.set_shape)

    :param tensor: Tensor to reshape.
    :param tensor_shape: Shape to apply to the tensor.
    :returns: A reshaped tensor.
    """
    # NOTE: That SOUND LIKE IN PLACE HERE ?
    tensor.set_shape(tensor_shape)
    return tensor
