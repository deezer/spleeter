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

import hashlib
import tarfile
import os

from tempfile import NamedTemporaryFile

import requests

from . import ModelProvider
from ...utils.logging import get_logger

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


def compute_file_checksum(path):
    """ Computes given path file sha256.

    :param path: Path of the file to compute checksum for.
    :returns: File checksum.
    """
    sha256 = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


class GithubModelProvider(ModelProvider):
    """ A ModelProvider implementation backed on Github for remote storage. """

    LATEST_RELEASE = 'v1.4.0'
    RELEASE_PATH = 'releases/download'
    CHECKSUM_INDEX = 'checksum.json'

    def __init__(self, host, repository, release):
        """ Default constructor.

        :param host: Host to the Github instance to reach.
        :param repository: Repository path within target Github.
        :param release: Release name to get models from.
        """
        self._host = host
        self._repository = repository
        self._release = release

    def checksum(self, name):
        """ Downloads and returns reference checksum for the given model name.

        :param name: Name of the model to get checksum for.
        :returns: Checksum of the required model.
        :raise ValueError: If the given model name is not indexed.
        """
        url = '{}/{}/{}/{}/{}'.format(
            self._host,
            self._repository,
            self.RELEASE_PATH,
            self._release,
            self.CHECKSUM_INDEX)
        response = requests.get(url)
        response.raise_for_status()
        index = response.json()
        if name not in index:
            raise ValueError('No checksum for model {}'.format(name))
        return index[name]

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
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            archive = NamedTemporaryFile(delete=False)
            try:
                with archive as stream:
                    # Note: check for chunk size parameters ?
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            stream.write(chunk)
                get_logger().info('Validating archive checksum')
                if compute_file_checksum(archive.name) != self.checksum(name):
                    raise IOError('Downloaded file is corrupted, please retry')
                get_logger().info('Extracting downloaded %s archive', name)
                with tarfile.open(name=archive.name) as tar:
                    tar.extractall(path=path)
            finally:
                os.unlink(archive.name)
        get_logger().info('%s model file(s) extracted', name)
