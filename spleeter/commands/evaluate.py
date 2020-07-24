#!/usr/bin/env python
# coding: utf8

"""
    Entrypoint provider for performing model evaluation.

    Evaluation is performed against musDB dataset.

    USAGE: python -m spleeter evaluate \
        -p /path/to/params \
        -o /path/to/output/dir \
        [-m] \
        --mus_dir /path/to/musdb dataset
"""

import sys
import json

from argparse import Namespace
from itertools import product
from glob import glob
from os.path import join, exists

# pylint: disable=import-error
import numpy as np
import pandas as pd
# pylint: enable=import-error

from .separate import entrypoint as separate_entrypoint
from ..utils.logging import get_logger

try:
    import musdb
    import museval
except ImportError:
    logger = get_logger()
    logger.error('Extra dependencies musdb and museval not found')
    logger.error('Please install musdb and museval first, abort')
    sys.exit(1)

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

_SPLIT = 'test'
_MIXTURE = 'mixture.wav'
_AUDIO_DIRECTORY = 'audio'
_METRICS_DIRECTORY = 'metrics'
_INSTRUMENTS = ('vocals', 'drums', 'bass', 'other')
_METRICS = ('SDR', 'SAR', 'SIR', 'ISR')


def _separate_evaluation_dataset(arguments, musdb_root_directory, params):
    """ Performs audio separation on the musdb dataset from
    the given directory and params.

    :param arguments: Entrypoint arguments.
    :param musdb_root_directory: Directory to retrieve dataset from.
    :param params: Spleeter configuration to apply to separation.
    :returns: Separation output directory path.
    """
    songs = glob(join(musdb_root_directory, _SPLIT, '*/'))
    mixtures = [join(song, _MIXTURE) for song in songs]
    audio_output_directory = join(
        arguments.output_path,
        _AUDIO_DIRECTORY)
    separate_entrypoint(
        Namespace(
            audio_adapter=arguments.audio_adapter,
            configuration=arguments.configuration,
            inputs=mixtures,
            output_path=join(audio_output_directory, _SPLIT),
            filename_format='{foldername}/{instrument}.{codec}',
            codec='wav',
            duration=600.,
            offset=0.,
            bitrate='128k',
            MWF=arguments.MWF,
            verbose=arguments.verbose,
            stft_backend=arguments.stft_backend),
        params)
    return audio_output_directory


def _compute_musdb_metrics(
        arguments,
        musdb_root_directory,
        audio_output_directory):
    """ Generates musdb metrics fro previsouly computed audio estimation.

    :param arguments: Entrypoint arguments.
    :param audio_output_directory: Directory to get audio estimation from.
    :returns: Path of generated metrics directory.
    """
    metrics_output_directory = join(
        arguments.output_path,
        _METRICS_DIRECTORY)
    get_logger().info('Starting musdb evaluation (this could be long) ...')
    dataset = musdb.DB(
        root=musdb_root_directory,
        is_wav=True,
        subsets=[_SPLIT])
    museval.eval_mus_dir(
        dataset=dataset,
        estimates_dir=audio_output_directory,
        output_dir=metrics_output_directory)
    get_logger().info('musdb evaluation done')
    return metrics_output_directory


def _compile_metrics(metrics_output_directory):
    """ Compiles metrics from given directory and returns
    results as dict.

    :param metrics_output_directory: Directory to get metrics from.
    :returns: Compiled metrics as dict.
    """
    songs = glob(join(metrics_output_directory, 'test/*.json'))
    index = pd.MultiIndex.from_tuples(
        product(_INSTRUMENTS, _METRICS),
        names=['instrument', 'metric'])
    pd.DataFrame([], index=['config1', 'config2'], columns=index)
    metrics = {
        instrument: {k: [] for k in _METRICS}
        for instrument in _INSTRUMENTS}
    for song in songs:
        with open(song, 'r') as stream:
            data = json.load(stream)
        for target in data['targets']:
            instrument = target['name']
            for metric in _METRICS:
                sdr_med = np.median([
                    frame['metrics'][metric]
                    for frame in target['frames']
                    if not np.isnan(frame['metrics'][metric])])
                metrics[instrument][metric].append(sdr_med)
    return metrics


def entrypoint(arguments, params):
    """ Command entrypoint.

    :param arguments: Command line parsed argument as argparse.Namespace.
    :param params: Deserialized JSON configuration file provided in CLI args.
    """
    # Parse and check musdb directory.
    musdb_root_directory = arguments.mus_dir
    if not exists(musdb_root_directory):
        raise IOError(f'musdb directory {musdb_root_directory} not found')
    # Separate musdb sources.
    audio_output_directory = _separate_evaluation_dataset(
        arguments,
        musdb_root_directory,
        params)
    # Compute metrics with musdb.
    metrics_output_directory = _compute_musdb_metrics(
        arguments,
        musdb_root_directory,
        audio_output_directory)
    # Compute and pretty print median metrics.
    metrics = _compile_metrics(metrics_output_directory)
    for instrument, metric in metrics.items():
        get_logger().info('%s:', instrument)
        for metric, value in metric.items():
            get_logger().info('%s: %s', metric, f'{np.median(value):.3f}')

    return metrics
