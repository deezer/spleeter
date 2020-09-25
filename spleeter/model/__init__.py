#!/usr/bin/env python
# coding: utf8

""" This package provide an estimator builder as well as model functions. """

import importlib

# pylint: disable=import-error
import tensorflow as tf

from tensorflow.signal import stft, inverse_stft, hann_window
# pylint: enable=import-error

from ..utils.tensor import pad_and_partition, pad_and_reshape

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


placeholder = tf.compat.v1.placeholder


def get_model_function(model_type):
    """
        Get tensorflow function of the model to be applied to the input tensor.
        For instance "unet.softmax_unet" will return the softmax_unet function
        in the "unet.py" submodule of the current module (spleeter.model).

        Params:
        - model_type: str
        the relative module path to the model function.

        Returns:
        A tensorflow function to be applied to the input tensor to get the
        multitrack output.
    """
    relative_path_to_module = '.'.join(model_type.split('.')[:-1])
    model_name = model_type.split('.')[-1]
    main_module = '.'.join((__name__, 'functions'))
    path_to_module = f'{main_module}.{relative_path_to_module}'
    module = importlib.import_module(path_to_module)
    model_function = getattr(module, model_name)
    return model_function


class InputProvider(object):

    def __init__(self, params):
        self.params = params

    def get_input_dict_placeholders(self):
        raise NotImplementedError()

    @property
    def input_names(self):
        raise NotImplementedError()

    def get_feed_dict(self, features, *args):
        raise NotImplementedError()


class WaveformInputProvider(InputProvider):

    @property
    def input_names(self):
        return ["audio_id", "waveform"]

    def get_input_dict_placeholders(self):
        shape = (None, self.params['n_channels'])
        features = {
            'waveform': placeholder(tf.float32, shape=shape, name="waveform"),
            'audio_id': placeholder(tf.string, name="audio_id")}
        return features

    def get_feed_dict(self, features, waveform, audio_id):
        return {features["audio_id"]: audio_id, features["waveform"]: waveform}


class SpectralInputProvider(InputProvider):

    def __init__(self, params):
        super().__init__(params)
        self.stft_input_name = "{}_stft".format(self.params["mix_name"])

    @property
    def input_names(self):
        return ["audio_id", self.stft_input_name]

    def get_input_dict_placeholders(self):
        features = {
            self.stft_input_name: placeholder(tf.complex64,
                                              shape=(None, self.params["frame_length"]//2+1,
                                                     self.params['n_channels']),
                                              name=self.stft_input_name),
            'audio_id': placeholder(tf.string, name="audio_id")}
        return features

    def get_feed_dict(self, features, stft, audio_id):
        return {features["audio_id"]: audio_id, features[self.stft_input_name]: stft}


class InputProviderFactory(object):

    @staticmethod
    def get(params):
        stft_backend = params["stft_backend"]
        assert stft_backend in ("tensorflow", "librosa"), "Unexpected backend {}".format(stft_backend)
        if stft_backend == "tensorflow":
            return WaveformInputProvider(params)
        else:
            return SpectralInputProvider(params)


class EstimatorSpecBuilder(object):
    """ A builder class that allows to builds a multitrack unet model
    estimator. The built model estimator has a different behaviour when
    used in a train/eval mode and in predict mode.

    * In train/eval mode:   it takes as input and outputs magnitude spectrogram
    * In predict mode:      it takes as input and outputs waveform. The whole
                            separation process is then done in this function
                            for performance reason: it makes it possible to run
                            the whole spearation process (including STFT and
                            inverse STFT) on GPU.

    :Example:

    >>> from spleeter.model import EstimatorSpecBuilder
    >>> builder = EstimatorSpecBuilder()
    >>> builder.build_predict_model()
    >>> builder.build_evaluation_model()
    >>> builder.build_train_model()

    >>> from spleeter.model import model_fn
    >>> estimator = tf.estimator.Estimator(model_fn=model_fn, ...)
    """

    # Supported model functions.
    DEFAULT_MODEL = 'unet.unet'

    # Supported loss functions.
    L1_MASK = 'L1_mask'
    WEIGHTED_L1_MASK = 'weighted_L1_mask'

    # Supported optimizers.
    ADADELTA = 'Adadelta'
    SGD = 'SGD'

    # Math constants.
    WINDOW_COMPENSATION_FACTOR = 2./3.
    EPSILON = 1e-10

    def __init__(self, features, params):
        """ Default constructor. Depending on built model
        usage, the provided features should be different:

        * In train/eval mode:   features is a dictionary with a
                                "mix_spectrogram" key, associated to the
                                mix magnitude spectrogram.
        * In predict mode:      features is a dictionary with a "waveform"
                                key, associated to the waveform of the sound
                                to be separated.

        :param features: The input features for the estimator.
        :param params: Some hyperparameters as a dictionary.
        """

        self._features = features
        self._params = params
        # Get instrument name.
        self._mix_name = params['mix_name']
        self._instruments = params['instrument_list']
        # Get STFT/signals parameters
        self._n_channels = params['n_channels']
        self._T = params['T']
        self._F = params['F']
        self._frame_length = params['frame_length']
        self._frame_step = params['frame_step']

    def include_stft_computations(self):
        return self._params["stft_backend"] == "tensorflow"

    def _build_model_outputs(self):
        """ Created a batch_sizexTxFxn_channels input tensor containing
        mix magnitude spectrogram, then an output dict from it according
        to the selected model in internal parameters.

        :returns: Build output dict.
        :raise ValueError: If required model_type is not supported.
        """

        input_tensor = self.spectrogram_feature
        model = self._params.get('model', None)
        if model is not None:
            model_type = model.get('type', self.DEFAULT_MODEL)
        else:
            model_type = self.DEFAULT_MODEL
        try:
            apply_model = get_model_function(model_type)
        except ModuleNotFoundError:
            raise ValueError(f'No model function {model_type} found')
        self._model_outputs = apply_model(
            input_tensor,
            self._instruments,
            self._params['model']['params'])

    def _build_loss(self, labels):
        """ Construct tensorflow loss and metrics

        :param output_dict: dictionary of network outputs (key: instrument
            name, value: estimated spectrogram of the instrument)
        :param labels: dictionary of target outputs (key: instrument
            name, value: ground truth spectrogram of the instrument)
        :returns: tensorflow (loss, metrics) tuple.
        """
        output_dict = self.model_outputs
        loss_type = self._params.get('loss_type', self.L1_MASK)
        if loss_type == self.L1_MASK:
            losses = {
                name: tf.reduce_mean(tf.abs(output - labels[name]))
                for name, output in output_dict.items()
            }
        elif loss_type == self.WEIGHTED_L1_MASK:
            losses = {
                name: tf.reduce_mean(
                    tf.reduce_mean(
                        labels[name],
                        axis=[1, 2, 3],
                        keep_dims=True) *
                    tf.abs(output - labels[name]))
                for name, output in output_dict.items()
            }
        else:
            raise ValueError(f"Unkwnown loss type: {loss_type}")
        loss = tf.reduce_sum(list(losses.values()))
        # Add metrics for monitoring each instrument.
        metrics = {k: tf.compat.v1.metrics.mean(v) for k, v in losses.items()}
        metrics['absolute_difference'] = tf.compat.v1.metrics.mean(loss)
        return loss, metrics

    def _build_optimizer(self):
        """ Builds an optimizer instance from internal parameter values.

        Default to AdamOptimizer if not specified.

        :returns: Optimizer instance from internal configuration.
        """
        name = self._params.get('optimizer')
        if name == self.ADADELTA:
            return tf.compat.v1.train.AdadeltaOptimizer()
        rate = self._params['learning_rate']
        if name == self.SGD:
            return tf.compat.v1.train.GradientDescentOptimizer(rate)
        return tf.compat.v1.train.AdamOptimizer(rate)

    @property
    def instruments(self):
        return self._instruments

    @property
    def stft_name(self):
        return f'{self._mix_name}_stft'

    @property
    def spectrogram_name(self):
        return f'{self._mix_name}_spectrogram'

    def _build_stft_feature(self):
        """ Compute STFT of waveform and slice the STFT in segment
         with the right length to feed the network.
        """

        stft_name = self.stft_name
        spec_name = self.spectrogram_name

        if stft_name not in self._features:
            # pad input with a frame of zeros
            waveform =  tf.concat([
                            tf.zeros((self._frame_length, self._n_channels)),
                            self._features['waveform']
                            ],
                            0
                        )
            stft_feature = tf.transpose(
                stft(
                    tf.transpose(waveform),
                    self._frame_length,
                    self._frame_step,
                    window_fn=lambda frame_length, dtype: (
                        hann_window(frame_length, periodic=True, dtype=dtype)),
                    pad_end=True),
                perm=[1, 2, 0])
            self._features[f'{self._mix_name}_stft'] = stft_feature
        if spec_name not in self._features:
            self._features[spec_name] = tf.abs(
                pad_and_partition(self._features[stft_name], self._T))[:, :, :self._F, :]

    @property
    def model_outputs(self):
        if not hasattr(self, "_model_outputs"):
            self._build_model_outputs()
        return self._model_outputs

    @property
    def outputs(self):
        if not hasattr(self, "_outputs"):
            self._build_outputs()
        return self._outputs

    @property
    def stft_feature(self):
        if self.stft_name not in self._features:
            self._build_stft_feature()
        return self._features[self.stft_name]

    @property
    def spectrogram_feature(self):
        if self.spectrogram_name not in self._features:
            self._build_stft_feature()
        return self._features[self.spectrogram_name]

    @property
    def masks(self):
        if not hasattr(self, "_masks"):
            self._build_masks()
        return self._masks

    @property
    def masked_stfts(self):
        if not hasattr(self, "_masked_stfts"):
            self._build_masked_stfts()
        return self._masked_stfts

    def _inverse_stft(self, stft_t, time_crop=None):
        """ Inverse and reshape the given STFT

        :param stft_t: input STFT
        :returns: inverse STFT (waveform)
        """
        inversed = inverse_stft(
            tf.transpose(stft_t, perm=[2, 0, 1]),
            self._frame_length,
            self._frame_step,
            window_fn=lambda frame_length, dtype: (
                hann_window(frame_length, periodic=True, dtype=dtype))
        ) * self.WINDOW_COMPENSATION_FACTOR
        reshaped = tf.transpose(inversed)
        if time_crop is None:
            time_crop = tf.shape(self._features['waveform'])[0]
        return reshaped[self._frame_length:self._frame_length+time_crop, :]

    def _build_mwf_output_waveform(self):
        """ Perform separation with multichannel Wiener Filtering using Norbert.
        Note: multichannel Wiener Filtering is not coded in Tensorflow and thus
        may be quite slow.

        :returns: dictionary of separated waveforms (key: instrument name,
            value: estimated waveform of the instrument)
        """
        import norbert  # pylint: disable=import-error
        output_dict = self.model_outputs
        x = self.stft_feature
        v = tf.stack(
            [
                pad_and_reshape(
                    output_dict[f'{instrument}_spectrogram'],
                    self._frame_length,
                    self._F)[:tf.shape(x)[0], ...]
                for instrument in self._instruments
            ],
            axis=3)
        input_args = [v, x]
        stft_function = tf.py_function(
            lambda v, x: norbert.wiener(v.numpy(), x.numpy()),
            input_args,
            tf.complex64),
        return {
            instrument: self._inverse_stft(stft_function[0][:, :, :, k])
            for k, instrument in enumerate(self._instruments)
        }

    def _extend_mask(self, mask):
        """ Extend mask, from reduced number of frequency bin to the number of
        frequency bin in the STFT.

        :param mask: restricted mask
        :returns: extended mask
        :raise ValueError: If invalid mask_extension parameter is set.
        """
        extension = self._params['mask_extension']
        # Extend with average
        # (dispatch according to energy in the processed band)
        if extension == "average":
            extension_row = tf.reduce_mean(mask, axis=2, keepdims=True)
        # Extend with 0
        # (avoid extension artifacts but not conservative separation)
        elif extension == "zeros":
            mask_shape = tf.shape(mask)
            extension_row = tf.zeros((
                mask_shape[0],
                mask_shape[1],
                1,
                mask_shape[-1]))
        else:
            raise ValueError(f'Invalid mask_extension parameter {extension}')
        n_extra_row = self._frame_length // 2 + 1 - self._F
        extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
        return tf.concat([mask, extension], axis=2)

    def _build_masks(self):
        """
        Compute masks from the output spectrograms of the model.
        :return:
        """
        output_dict = self.model_outputs
        stft_feature = self.stft_feature
        separation_exponent = self._params['separation_exponent']
        output_sum = tf.reduce_sum(
            [e ** separation_exponent for e in output_dict.values()],
            axis=0
        ) + self.EPSILON
        out = {}
        for instrument in self._instruments:
            output = output_dict[f'{instrument}_spectrogram']
            # Compute mask with the model.
            instrument_mask = (output ** separation_exponent
                               + (self.EPSILON / len(output_dict))) / output_sum
            # Extend mask;
            instrument_mask = self._extend_mask(instrument_mask)
            # Stack back mask.
            old_shape = tf.shape(instrument_mask)
            new_shape = tf.concat(
                [[old_shape[0] * old_shape[1]], old_shape[2:]],
                axis=0)
            instrument_mask = tf.reshape(instrument_mask, new_shape)
            # Remove padded part (for mask having the same size as STFT);

            instrument_mask = instrument_mask[
                              :tf.shape(stft_feature)[0], ...]
            out[instrument] = instrument_mask
        self._masks = out

    def _build_masked_stfts(self):
        input_stft = self.stft_feature
        out = {}
        for instrument, mask in self.masks.items():
            out[instrument] = tf.cast(mask, dtype=tf.complex64) * input_stft
        self._masked_stfts = out

    def _build_manual_output_waveform(self, masked_stft):
        """ Perform ratio mask separation

        :param output_dict: dictionary of estimated spectrogram (key: instrument
            name, value: estimated spectrogram of the instrument)
        :returns: dictionary of separated waveforms (key: instrument name,
            value: estimated waveform of the instrument)
        """

        output_waveform = {}
        for instrument, stft_data in masked_stft.items():
            output_waveform[instrument] = self._inverse_stft(stft_data)
        return output_waveform

    def _build_output_waveform(self, masked_stft):
        """ Build output waveform from given output dict in order to be used in
        prediction context. Regarding of the configuration building method will
        be using MWF.

        :returns: Built output waveform.
        """

        if self._params.get('MWF', False):
            output_waveform = self._build_mwf_output_waveform()
        else:
            output_waveform = self._build_manual_output_waveform(masked_stft)
        return output_waveform

    def _build_outputs(self):
        if self.include_stft_computations():
            self._outputs = self._build_output_waveform(self.masked_stfts)
        else:
            self._outputs = self.masked_stfts

        if 'audio_id' in self._features:
            self._outputs['audio_id'] = self._features['audio_id']

    def build_predict_model(self):
        """ Builder interface for creating model instance that aims to perform
        prediction / inference over given track. The output of such estimator
        will be a dictionary with a "<instrument>" key per separated instrument
        , associated to the estimated separated waveform of the instrument.

        :returns: An estimator for performing prediction.
        """

        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.PREDICT,
            predictions=self.outputs)

    def build_evaluation_model(self, labels):
        """ Builder interface for creating model instance that aims to perform
        model evaluation. The output of such estimator will be a dictionary
        with a key "<instrument>_spectrogram" per separated instrument,
        associated to the estimated separated instrument magnitude spectrogram.

        :param labels: Model labels.
        :returns: An estimator for performing model evaluation.
        """
        loss, metrics = self._build_loss(labels)
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=metrics)

    def build_train_model(self, labels):
        """ Builder interface for creating model instance that aims to perform
        model training. The output of such estimator will be a dictionary
        with a key "<instrument>_spectrogram" per separated instrument,
        associated to the estimated separated instrument magnitude spectrogram.

        :param labels: Model labels.
        :returns: An estimator for performing model training.
        """
        loss, metrics = self._build_loss(labels)
        optimizer = self._build_optimizer()
        train_operation = optimizer.minimize(
                loss=loss,
                global_step=tf.compat.v1.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_operation,
            eval_metric_ops=metrics,
        )


def model_fn(features, labels, mode, params, config):
    """

    :param features:
    :param labels: 
    :param mode: Estimator mode.
    :param params: 
    :param config: TF configuration (not used).
    :returns: Built EstimatorSpec.
    :raise ValueError: If estimator mode is not supported.
    """
    builder = EstimatorSpecBuilder(features, params)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return builder.build_predict_model()
    elif mode == tf.estimator.ModeKeys.EVAL:
        return builder.build_evaluation_model(labels)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        return builder.build_train_model(labels)
    raise ValueError(f'Unknown mode {mode}')
