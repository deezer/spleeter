#!/usr/bin/env python
# coding: utf8

""" Utility functions for creating estimator. """

from pathlib import Path
from os.path import join
from tempfile import gettempdir

# pylint: disable=import-error
import tensorflow as tf

from tensorflow.contrib import predictor
# pylint: enable=import-error

from ..model import model_fn
from ..model.provider import get_default_model_provider

# Default exporting directory for predictor.
DEFAULT_EXPORT_DIRECTORY = join(gettempdir(), 'serving')


def create_estimator(params, MWF):
    """
        Initialize tensorflow estimator that will perform separation

        Params:
        - params: a dictionnary of parameters for building the model

        Returns:
            a tensorflow estimator
    """
    # Load model.
    model_directory = params['model_dir']
    model_provider = get_default_model_provider()
    params['model_dir'] = model_provider.get(model_directory)
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


def to_predictor(estimator, directory=DEFAULT_EXPORT_DIRECTORY):
    """ Exports given estimator as predictor into the given directory
    and returns associated tf.predictor instance.

    :param estimator: Estimator to export.
    :param directory: (Optional) path to write exported model into.
    """
    def receiver():
        shape = (None, estimator.params['n_channels'])
        features = {
            'waveform': tf.compat.v1.placeholder(tf.float32, shape=shape),
            'audio_id': tf.compat.v1.placeholder(tf.string)}
        return tf.estimator.export.ServingInputReceiver(features, features)

    estimator.export_saved_model(directory, receiver)
    versions = [
        model for model in Path(directory).iterdir()
        if model.is_dir() and 'temp' not in str(model)]
    latest = str(sorted(versions)[-1])
    return predictor.from_saved_model(latest)
