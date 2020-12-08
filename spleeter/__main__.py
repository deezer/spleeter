#!/usr/bin/env python
# coding: utf8

"""
    Python oneliner script usage.

    USAGE: python -m spleeter {train,evaluate,separate} ...
"""

import json

from functools import partial
from itertools import product
from glob import glob
from os.path import join
from pathlib import Path
from typing import Any, Container, Dict, List

from . import SpleeterError
from .audio import Codec
from .audio.adapter import AudioAdapter
from .options import *
from .dataset import get_training_dataset, get_validation_dataset
from .model import model_fn
from .model.provider import ModelProvider
from .separator import Separator
from .utils.configuration import load_configuration
from .utils.logging import configure_logger, logger

# pyright: reportMissingImports=false
# pylint: disable=import-error
import numpy as np
import pandas as pd
import tensorflow as tf

from typer import Exit, Typer
# pylint: enable=import-error

spleeter: Typer = Typer()
""" """


@spleeter.command()
def train(
        adapter: str = AudioAdapterOption,
        data: Path = TrainingDataDirectoryOption,
        params_filename: str = ModelParametersOption,
        verbose: bool = VerboseOption) -> None:
    """
        Train a source separation model
    """
    configure_logger(verbose)
    audio_adapter = AudioAdapter.get(adapter)
    audio_path = str(data)
    params = load_configuration(params_filename)
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
    input_fn = partial(get_training_dataset, params, audio_adapter, audio_path)
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn,
        max_steps=params['train_max_steps'])
    input_fn = partial(
        get_validation_dataset,
        params,
        audio_adapter,
        audio_path)
    evaluation_spec = tf.estimator.EvalSpec(
        input_fn=input_fn,
        steps=None,
        throttle_secs=params['throttle_secs'])
    logger.info('Start model training')
    tf.estimator.train_and_evaluate(estimator, train_spec, evaluation_spec)
    ModelProvider.writeProbe(params['model_dir'])
    logger.info('Model training done')


@spleeter.commmand()
def separate(
        files: List[Path] = AudioInputArgument,
        adapter: str = AudioAdapterOption,
        bitrate: str = AudioBitrateOption,
        codec: Codec = AudioCodecOption,
        duration: float = AudioDurationOption,
        offset: float = AudioOffsetOption,
        output_path: Path = AudioAdapterOption,
        stft_backend: STFTBackend = AudioSTFTBackendOption,
        filename_format: str = FilenameFormatOption,
        params_filename: str = ModelParametersOption,
        mwf: bool = MWFOption,
        verbose: bool = VerboseOption) -> None:
    """
        Separate audio file(s)
    """
    configure_logger(verbose)
    audio_adapter: AudioAdapter = AudioAdapter.get(adapter)
    separator: Separator = Separator(
        params_filename,
        MWF=mwf,
        stft_backend=stft_backend)
    for filename in files:
        separator.separate_to_file(
            filename,
            output_path,
            audio_adapter=audio_adapter,
            offset=offset,
            duration=duration,
            codec=codec,
            bitrate=bitrate,
            filename_format=filename_format,
            synchronous=False)
    separator.join()


EVALUATION_SPLIT: str = 'test'
EVALUATION_METRICS_DIRECTORY: str = 'metrics'
EVALUATION_INSTRUMENTS: Container[str] = ('vocals', 'drums', 'bass', 'other')
EVALUATION_METRICS: Container[str] = ('SDR', 'SAR', 'SIR', 'ISR')
EVALUATION_MIXTURE: str = 'mixture.wav'
EVALUATION_AUDIO_DIRECTORY: str = 'audio'


def _compile_metrics(metrics_output_directory) -> Dict:
    """
        Compiles metrics from given directory and returns results as dict.

        Parameters:
            metrics_output_directory (str):
                Directory to get metrics from.

        Returns:
            Dict:
                Compiled metrics as dict.
    """
    songs = glob(join(metrics_output_directory, 'test/*.json'))
    index = pd.MultiIndex.from_tuples(
        product(EVALUATION_INSTRUMENTS, EVALUATION_METRICS),
        names=['instrument', 'metric'])
    pd.DataFrame([], index=['config1', 'config2'], columns=index)
    metrics = {
        instrument: {k: [] for k in EVALUATION_METRICS}
        for instrument in EVALUATION_INSTRUMENTS}
    for song in songs:
        with open(song, 'r') as stream:
            data = json.load(stream)
        for target in data['targets']:
            instrument = target['name']
            for metric in EVALUATION_METRICS:
                sdr_med = np.median([
                    frame['metrics'][metric]
                    for frame in target['frames']
                    if not np.isnan(frame['metrics'][metric])])
                metrics[instrument][metric].append(sdr_med)
    return metrics


@spleeter.command()
def evaluate(
        adapter: str = AudioAdapterOption,
        output_path: Path = AudioAdapterOption,
        stft_backend: STFTBackend = AudioSTFTBackendOption,
        params_filename: str = ModelParametersOption,
        mus_dir: Path = MUSDBDirectoryOption,
        mwf: bool = MWFOption,
        verbose: bool = VerboseOption) -> Dict:
    """
        Evaluate a model on the musDB test dataset
    """
    configure_logger(verbose)
    try:
        import musdb
        import museval
    except ImportError:
        logger.error('Extra dependencies musdb and museval not found')
        logger.error('Please install musdb and museval first, abort')
        raise Exit(10)
    # Separate musdb sources.
    songs = glob(join(mus_dir, EVALUATION_SPLIT, '*/'))
    mixtures = [join(song, EVALUATION_MIXTURE) for song in songs]
    audio_output_directory = join(output_path, EVALUATION_AUDIO_DIRECTORY)
    separate(
        adapter=adapter,
        params_filename=params_filename,
        files=mixtures,
        output_path=output_path,
        filename_format='{foldername}/{instrument}.{codec}',
        codec=Codec.WAV,
        mwf=mwf,
        verbose=verbose,
        stft_backend=stft_backend)
    # Compute metrics with musdb.
    metrics_output_directory = join(output_path, EVALUATION_METRICS_DIRECTORY)
    logger.info('Starting musdb evaluation (this could be long) ...')
    dataset = musdb.DB(root=mus_dir, is_wav=True, subsets=[EVALUATION_SPLIT])
    museval.eval_mus_dir(
        dataset=dataset,
        estimates_dir=audio_output_directory,
        output_dir=metrics_output_directory)
    logger.info('musdb evaluation done')
    # Compute and pretty print median metrics.
    metrics = _compile_metrics(metrics_output_directory)
    for instrument, metric in metrics.items():
        logger.info(f'{instrument}:')
        for metric, value in metric.items():
            logger.info(f'{metric}: {np.median(value):.3f}')
    return metrics


if __name__ == '__main__':
    try:
        spleeter()
    except SpleeterError as e:
        logger.error(e)
