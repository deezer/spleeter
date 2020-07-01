#!/usr/bin/env python
# coding: utf8

""" Utility functions for creating estimator. """

from pathlib import Path
from os.path import join

# pylint: disable=import-error
import tensorflow as tf


from ..model import model_fn
from ..model.provider import get_default_model_provider



def get_default_model_dir(model_dir):
    """
    Transforms a string like 'spleeter:2stems' into an actual path.
    :param model_dir:
    :return:
    """
    model_provider = get_default_model_provider()
    return model_provider.get(model_dir)

def create_estimator(params, MWF):
    """
        Initialize tensorflow estimator that will perform separation

        Params:
        - params: a dictionary of parameters for building the model

        Returns:
            a tensorflow estimator
    """
    # Load model.


    params['model_dir'] = get_default_model_dir(params['model_dir'])
    params['MWF'] = MWF
    # Setup config
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config = tf.estimator.RunConfig(session_config=session_config)
    # Setup estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params['model_dir'],
        params=params,
        config=config
    )
    return estimator
