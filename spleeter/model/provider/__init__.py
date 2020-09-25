#!/usr/bin/env python
# coding: utf8

"""
    This package provides tools for downloading model from network
    using remote storage abstraction.

    :Example:

    >>> provider = MyProviderImplementation()
    >>> provider.get('/path/to/local/storage', params)
"""

from abc import ABC, abstractmethod
from os import environ, makedirs
from os.path import exists, isabs, join, sep

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


class ModelProvider(ABC):
    """
        A ModelProvider manages model files on disk and
        file download is not available.
    """

    DEFAULT_MODEL_PATH = environ.get('MODEL_PATH', 'pretrained_models')
    MODEL_PROBE_PATH = '.probe'

    @abstractmethod
    def download(self, name, path):
        """ Download model denoted by the given name to disk.

        :param name: Name of the model to download.
        :param path: Path of the directory to save model into.
        """
        pass

    @staticmethod
    def writeProbe(directory):
        """ Write a model probe file into the given directory.

        :param directory: Directory to write probe into.
        """
        probe = join(directory, ModelProvider.MODEL_PROBE_PATH)
        with open(probe, 'w') as stream:
            stream.write('OK')

    def get(self, model_directory):
        """ Ensures required model is available at given location.

        :param model_directory: Expected model_directory to be available.
        :raise IOError: If model can not be retrieved.
        """
        # Expend model directory if needed.
        if not isabs(model_directory):
            model_directory = join(self.DEFAULT_MODEL_PATH, model_directory)
        # Download it if not exists.
        model_probe = join(model_directory, self.MODEL_PROBE_PATH)
        if not exists(model_probe):
            if not exists(model_directory):
                makedirs(model_directory)
                self.download(
                    model_directory.split(sep)[-1],
                    model_directory)
                self.writeProbe(model_directory)
        return model_directory


def get_default_model_provider():
    """ Builds and returns a default model provider.

    :returns: A default model provider instance to use.
    """
    from .github import GithubModelProvider
    host = environ.get('GITHUB_HOST', 'https://github.com')
    repository = environ.get('GITHUB_REPOSITORY', 'deezer/spleeter')
    release = environ.get('GITHUB_RELEASE', GithubModelProvider.LATEST_RELEASE)
    return GithubModelProvider(host, repository, release)
