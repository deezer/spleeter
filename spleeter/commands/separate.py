#!/usr/bin/env python
# coding: utf8

"""
    Entrypoint provider for performing source separation.

    USAGE: python -m spleeter separate \
        -p /path/to/params \
        -i inputfile1 inputfile2 ... inputfilen
        -o /path/to/output/dir \
        -i /path/to/audio1.wav /path/to/audio2.mp3
"""

from multiprocessing import Pool
from os.path import isabs, join, split, splitext
from tempfile import gettempdir

# pylint: disable=import-error
import tensorflow as tf
import numpy as np
# pylint: enable=import-error

from ..utils.audio.adapter import get_audio_adapter
from ..utils.audio.convertor import to_n_channels
from ..utils.estimator import create_estimator
from ..utils.tensor import set_tensor_shape

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


def get_dataset(audio_adapter, filenames_and_crops, sample_rate, n_channels):
    """"
        Build a tensorflow dataset of waveform from a filename list wit crop
        information.

        Params:
        - audio_adapter:        An AudioAdapter instance to load audio from.
        - filenames_and_crops:  list of (audio_filename, start, duration)
                                tuples separation is performed on each filaneme
                                from start (in seconds) to start + duration
                                (in seconds).
        - sample_rate:          audio sample_rate of the input and output audio
                                signals
        - n_channels:           int, number of channels of the input and output
                                audio signals

        Returns
        A tensorflow dataset of waveform to feed a tensorflow estimator in
        predict mode.
    """
    filenames, starts, ends = list(zip(*filenames_and_crops))
    dataset = tf.data.Dataset.from_tensor_slices({
        'audio_id': list(filenames),
        'start': list(starts),
        'end': list(ends)
    })
    # Load waveform.
    dataset = dataset.map(
        lambda sample: dict(
            sample,
            **audio_adapter.load_tf_waveform(
                sample['audio_id'],
                sample_rate=sample_rate,
                offset=sample['start'],
                duration=sample['end'] - sample['start'])),
        num_parallel_calls=2)
    # Filter out error.
    dataset = dataset.filter(
        lambda sample: tf.logical_not(sample['waveform_error']))
    # Convert waveform to the right number of channels.
    dataset = dataset.map(
        lambda sample: dict(
            sample,
            waveform=to_n_channels(sample['waveform'], n_channels)))
    # Set number of channels (required for the model).
    dataset = dataset.map(
        lambda sample: dict(
            sample,
            waveform=set_tensor_shape(sample['waveform'], (None, n_channels))))
    return dataset


def process_audio(
        audio_adapter,
        filenames_and_crops, estimator, output_path,
        sample_rate, n_channels, codec, output_naming):
    """
        Perform separation on a list of audio ids.

        Params:
        - audio_adapter:        Audio adapter to use for audio I/O.
        - filenames_and_crops:  list of (audio_filename, start, duration)
                                tuples separation is performed on each filaneme
                                from start (in seconds) to start + duration
                                (in seconds).
        - estimator:            the tensorflow estimator that performs the
                                source separation.
        - output_path:          output_path where to export separated files.
        - sample_rate:          audio sample_rate of the input and output audio
                                signals
        - n_channels:           int, number of channels of the input and output
                                audio signals
        - codec:                string codec to be used for export (could be
                                "wav", "mp3", "ogg", "m4a") could be anything
                                supported by ffmpeg.
        - output_naming: string (= "filename" of "directory")
            naming convention for output.
            for an input file /path/to/audio/input_file.wav:
                * if output_naming is equal to "filename":
        output files will be put in the directory <output_path>/input_file
        (<output_path>/input_file/<instrument1>.<codec>,
         <output_path>/input_file/<instrument2>.<codec>...).
                * if output_naming is equal to "directory":
        output files will be put in the directory <output_path>/audio/
        (<output_path>/audio/<instrument1>.<codec>,
         <output_path>/audio/<instrument2>.<codec>...)
        Use "directory" when separating the MusDB dataset.

    """
    # Get estimator
    prediction = estimator.predict(
        lambda: get_dataset(
            audio_adapter,
            filenames_and_crops,
            sample_rate,
            n_channels),
        yield_single_examples=False)
    # initialize pool for audio export
    pool = Pool(16)
    tasks = []
    for sample in prediction:
        sample_filename = sample.pop('audio_id', 'unknown_filename').decode()
        input_directory, input_filename = split(sample_filename)
        if output_naming == 'directory':
            output_dirname = split(input_directory)[1]
        elif output_naming == 'filename':
            output_dirname = splitext(input_filename)[0]
        else:
            raise ValueError(f'Unknown output naming {output_naming}')
        for instrument, waveform in sample.items():
            filename = join(
                output_path,
                output_dirname,
                f'{instrument}.{codec}')
            tasks.append(
                pool.apply_async(
                    audio_adapter.save,
                    (filename, waveform, sample_rate, codec)))
    # Wait for everything to be written
    for task in tasks:
        task.wait(timeout=20)


def entrypoint(arguments, params):
    """ Command entrypoint.

    :param arguments: Command line parsed argument as argparse.Namespace.
    :param params: Deserialized JSON configuration file provided in CLI args.
    """
    audio_adapter = get_audio_adapter(arguments.audio_adapter)
    filenames = arguments.audio_filenames
    output_path = arguments.output_path
    max_duration = arguments.max_duration
    audio_codec = arguments.audio_codec
    output_naming = arguments.output_naming
    estimator = create_estimator(params, arguments.MWF)
    filenames_and_crops = [
        (filename, 0., max_duration)
        for filename in filenames]
    process_audio(
        audio_adapter,
        filenames_and_crops,
        estimator,
        output_path,
        params['sample_rate'],
        params['n_channels'],
        codec=audio_codec,
        output_naming=output_naming)
