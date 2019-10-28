#!/usr/bin/env python
# coding: utf8

"""
    A ModelProvider backed by Github Release feature.

    :Example:

    >>> from spleeter.model.provider import github
    >>> provider = github.GithubModelProvider(
            'github.com',
            'Deezer/spleeter',
            'latest')
    >>> provider.download('2stems', '/path/to/local/storage')
"""

import tarfile

from os import environ
from tempfile import TemporaryFile
from shutil import copyfileobj

import requests

from . import ModelProvider
from ...utils.logging import get_logger

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


class GithubModelProvider(ModelProvider):
    """ A ModelProvider implementation backed on Github for remote storage. """

    LATEST_RELEASE = 'v1.4.0'
    RELEASE_PATH = 'releases/download'

    def __init__(self, host, repository, release):
        """ Default constructor.

        :param host: Host to the Github instance to reach.
        :param repository: Repository path within target Github.
        :param release: Release name to get models from.
        """
        self._host = host
        self._repository = repository
        self._release = release

    def download(self, name, path):
        """ Download model denoted by the given name to disk.

        :param name: Name of the model to download.
        :param path: Path of the directory to save model into.
        """
        url = '{}/{}/{}/{}/{}.tar.gz'.format(
            self._host,
            self._repository,
            self.RELEASE_PATH,
            self._release,
            name)
        get_logger().info('Downloading model archive %s', url)
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise IOError(f'Resource {url} not found')
        with TemporaryFile() as stream:
            copyfileobj(response.raw, stream)
            get_logger().debug('Extracting downloaded archive')
            stream.seek(0)
            tar = tarfile.open(fileobj=stream)
            tar.extractall(path=path)
            tar.close()
        get_logger().debug('Model file extracted')
