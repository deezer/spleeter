#!/usr/bin/env python
# coding: utf8

""" Module that provides configuration loading function. """

import importlib.resources as loader
import json
from os.path import exists
from typing import Dict

from .. import SpleeterError, resources

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

_EMBEDDED_CONFIGURATION_PREFIX: str = "spleeter:"


def load_configuration(descriptor: str) -> Dict:
    """
    Load configuration from the given descriptor.
    Could be either a `spleeter:` prefixed embedded configuration name
    or a file system path to read configuration from.

    Parameters:
        descriptor (str):
            Configuration descriptor to use for lookup.

    Returns:
        Dict:
            Loaded description as dict.

    Raises:
        ValueError:
            If required embedded configuration does not exists.
        SpleeterError:
            If required configuration file does not exists.
    """
    # Embedded configuration reading.
    if descriptor.startswith(_EMBEDDED_CONFIGURATION_PREFIX):
        name = descriptor[len(_EMBEDDED_CONFIGURATION_PREFIX) :]
        if not loader.is_resource(resources, f"{name}.json"):
            raise SpleeterError(f"No embedded configuration {name} found")
        with loader.open_text(resources, f"{name}.json") as stream:
            return json.load(stream)
    # Standard file reading.
    if not exists(descriptor):
        raise SpleeterError(f"Configuration file {descriptor} not found")
    with open(descriptor, "r") as stream:
        return json.load(stream)
