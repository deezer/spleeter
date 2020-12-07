#!/usr/bin/env python
# coding: utf8

""" TO DOCUMENT """


from functools import partial
from pathlib import Path
from typing import List

from .audio import Codec
from .audio.adapter import AudioAdapter
from .options import *
from .dataset import get_training_dataset, get_validation_dataset
from .model import model_fn
from .model.provider import ModelProvider
from .separator import Separator
from .utils.configuration import load_configuration
from .utils.logging import get_logger


# pyright: reportMissingImports=false
# pylint: disable=import-error
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
    # TODO: try / catch or custom decorator for function handling.
    # TODO: handle verbose flag ?
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
    get_logger().info('Start model training')
    tf.estimator.train_and_evaluate(estimator, train_spec, evaluation_spec)
    ModelProvider.writeProbe(params['model_dir'])
    get_logger().info('Model training done')

_SPLIT = 'test'
_MIXTURE = 'mixture.wav'
_AUDIO_DIRECTORY = 'audio'
_METRICS_DIRECTORY = 'metrics'
_INSTRUMENTS = ('vocals', 'drums', 'bass', 'other')
_METRICS = ('SDR', 'SAR', 'SIR', 'ISR')


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
    try:
        import musdb
        import museval
    except ImportError:
        logger = get_logger()
        logger.error('Extra dependencies musdb and museval not found')
        logger.error('Please install musdb and museval first, abort')
        raise Exit(10)
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


@spleeter.command()
def evaluate(
        adapter: str = AudioAdapterOption,
        output_path: Path = AudioAdapterOption,
        stft_backend: STFTBackend = AudioSTFTBackendOption,
        params_filename: str = ModelParametersOption,
        mus_dir: Path = MUSDBDirectoryOption,
        mwf: bool = MWFOption,
        verbose: bool = VerboseOption) -> None:
    """
        Evaluate a model on the musDB test dataset
    """
    # Separate musdb sources.
    audio_output_directory = _separate_evaluation_dataset(
        arguments,
        mus_dir,
        params)
   # Compute metrics with musdb.
    metrics_output_directory = _compute_musdb_metrics(
        arguments,
        mus_dir,
        audio_output_directory)
    # Compute and pretty print median metrics.
    metrics = _compile_metrics(metrics_output_directory)
    for instrument, metric in metrics.items():
        get_logger().info('%s:', instrument)
        for metric, value in metric.items():
            get_logger().info('%s: %s', metric, f'{np.median(value):.3f}')
    return metrics


@spleeter.commmand()
def separate(
        adapter: str = AudioAdapterOption,
        bitrate: str = AudioBitrateOption,
        codec: Codec = AudioCodecOption,
        duration: float = AudioDurationOption,
        files: List[Path] = AudioInputArgument,
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
    # TODO: try / catch or custom decorator for function handling.
    # TODO: enable_logging()
    # TODO: handle MWF
    if verbose:
        # TODO: enable_tensorflow_logging()
        pass
    # PREV: params = load_configuration(arguments.configuration)
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


if __name__ == '__main__':
    # TODO: warnings.filterwarnings('ignore')
    spleeter()
