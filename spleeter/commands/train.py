#!/usr/bin/env python
# coding: utf8

"""
    Entrypoint provider for performing model training.

    USAGE: python -m spleeter train -p /path/to/params
"""

from functools import partial

# pylint: disable=import-error
import tensorflow as tf
# pylint: enable=import-error

from ..audio.adapter import get_audio_adapter
from ..dataset import get_training_dataset, get_validation_dataset
from ..model import model_fn
from ..model.provider import ModelProvider
from ..utils.logging import get_logger

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


def _create_estimator(params):
    """ Creates estimator.

    :param params: TF params to build estimator from.
    :returns: Built estimator.
    """
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params['model_dir'],
        params=params,
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=params['save_checkpoints_steps'],
            tf_random_seed=params['random_seed'],
            save_summary_steps=params['save_summary_steps'],
            session_config=session_config,
            log_step_count_steps=10,
            keep_checkpoint_max=2))
    return estimator


def _create_train_spec(params, audio_adapter, audio_path):
    """ Creates train spec.

    :param params: TF params to build spec from.
    :returns: Built train spec.
    """
    input_fn = partial(get_training_dataset, params, audio_adapter, audio_path)
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn,
        max_steps=params['train_max_steps'])
    return train_spec


def _create_evaluation_spec(params, audio_adapter, audio_path):
    """ Setup eval spec evaluating ever n seconds

    :param params: TF params to build spec from.
    :returns: Built evaluation spec.
    """
    input_fn = partial(
        get_validation_dataset,
        params,
        audio_adapter,
        audio_path)
    evaluation_spec = tf.estimator.EvalSpec(
        input_fn=input_fn,
        steps=None,
        throttle_secs=params['throttle_secs'])
    return evaluation_spec


def entrypoint(arguments, params):
    """ Command entrypoint.

    :param arguments: Command line parsed argument as argparse.Namespace.
    :param params: Deserialized JSON configuration file provided in CLI args.
    """
    audio_adapter = get_audio_adapter(arguments.audio_adapter)
    audio_path = arguments.audio_path
    estimator = _create_estimator(params)
    train_spec = _create_train_spec(params, audio_adapter, audio_path)
    evaluation_spec = _create_evaluation_spec(
        params,
        audio_adapter,
        audio_path)
    get_logger().info('Start model training')
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        evaluation_spec)
    ModelProvider.writeProbe(params['model_dir'])
    get_logger().info('Model training done')
