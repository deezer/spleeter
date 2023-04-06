#!/usr/bin/env python
# coding: utf8

"""
A ModelProvider backed by Github Release feature.

Examples:

```python
>>> from spleeter.model.provider import github
>>> provider = github.GithubModelProvider(
        'github.com',
        'Deezer/spleeter',
        'latest')
>>> provider.download('2stems', '/path/to/local/storage')
```
"""

import hashlib
import os
import tarfile
from os import environ
from tempfile import NamedTemporaryFile
from typing import Dict

# pyright: reportMissingImports=false
# pylint: disable=import-error
import httpx

from ...utils.logging import logger
from . import ModelProvider

# pylint: enable=import-error

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


def compute_file_checksum(path):
    """
    Computes given path file sha256.

    Parameters:
        path (str):
            Path of the file to compute checksum for.

    Returns:
        str:
            File checksum.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class GithubModelProvider(ModelProvider):
    """A ModelProvider implementation backed on Github for remote storage."""

    DEFAULT_HOST: str = "https://github.com"
    DEFAULT_REPOSITORY: str = "deezer/spleeter"

    CHECKSUM_INDEX: str = "checksum.json"
    LATEST_RELEASE: str = "v1.4.0"
    RELEASE_PATH: str = "releases/download"

    def __init__(self, host: str, repository: str, release: str) -> None:
        """Default constructor.

        Parameters:
            host (str):
                Host to the Github instance to reach.
            repository (str):
                Repository path within target Github.
            release (str):
                Release name to get models from.
        """
        self._host: str = host
        self._repository: str = repository
        self._release: str = release

    @classmethod
    def from_environ(cls) -> "GithubModelProvider":
        """
        Factory method that creates provider from envvars.

        Returns:
            GithubModelProvider:
                Created instance.
        """
        return cls(
            environ.get("GITHUB_HOST", cls.DEFAULT_HOST),
            environ.get("GITHUB_REPOSITORY", cls.DEFAULT_REPOSITORY),
            environ.get("GITHUB_RELEASE", cls.LATEST_RELEASE),
        )

    def checksum(self, name: str) -> str:
        """
        Downloads and returns reference checksum for the given model name.

        Parameters:
            name (str):
                Name of the model to get checksum for.

        Returns:
            str:
                Checksum of the required model.

        Raises:
            ValueError:
                If the given model name is not indexed.
        """
        url: str = "/".join(
            (
                self._host,
                self._repository,
                self.RELEASE_PATH,
                self._release,
                self.CHECKSUM_INDEX,
            )
        )
        response: httpx.Response = httpx.get(url)
        response.raise_for_status()
        index: Dict = response.json()
        if name not in index:
            raise ValueError(f"No checksum for model {name}")
        return index[name]

    def download(self, name: str, path: str) -> None:
        """
        Download model denoted by the given name to disk.

        Parameters:
            name (str):
                Name of the model to download.
            path (str):
                Path of the directory to save model into.
        """
        url: str = "/".join(
            (self._host, self._repository, self.RELEASE_PATH, self._release, name)
        )
        url = f"{url}.tar.gz"
        logger.info(f"Downloading model archive {url}")
        with httpx.Client(http2=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                archive = NamedTemporaryFile(delete=False)
                try:
                    with archive as stream:
                        for chunk in response.iter_raw():
                            stream.write(chunk)
                    logger.info("Validating archive checksum")
                    checksum: str = compute_file_checksum(archive.name)
                    if checksum != self.checksum(name):
                        raise IOError("Downloaded file is corrupted, please retry")
                    logger.info(f"Extracting downloaded {name} archive")
                    with tarfile.open(name=archive.name) as tar:
                        tar.extractall(path=path)
                finally:
                    os.unlink(archive.name)
        logger.info(f"{name} model file(s) extracted")
