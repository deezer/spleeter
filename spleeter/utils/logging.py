#!/usr/bin/env python
# coding: utf8

""" Centralized logging facilities for Spleeter. """

import logging
import warnings

from os import environ

# pyright: reportMissingImports=false
# pylint: disable=import-error
from tensorflow.compat.v1 import logging as tflogging
# pylint: enable=import-error

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger: logging.Logger = logging.getLogger('spleeter')
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def configure_logger(verbose: bool) -> None:
    """
        Configure application logger.

        Parameters:
            verbose (bool):
                `True` to use verbose logger, `False` otherwise.
    """
    if verbose:
        environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tflogging.set_verbosity(tflogging.INFO)
        logger.setLevel(logging.DEBUG)
    else:
        warnings.filterwarnings('ignore')
        environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tflogging.set_verbosity(tflogging.ERROR)
