#!/usr/bin/env python
# coding: utf8

""" Centralized logging facilities for Spleeter. """

import logging
import warnings

from os import environ

# pyright: reportMissingImports=false
# pylint: disable=import-error
from tensorflow.compat.v1 import logging as tf_logging
from typer import echo
# pylint: enable=import-error

__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


class TyperLoggerHandler(logging.Handler):
    """ A custom logger handler that use Typer echo. """

    def emit(self, record: logging.LogRecord) -> None:
        echo(self.format(record))


formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
handler = TyperLoggerHandler()
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
    tf_logger = tf_logging._get_logger()
    tf_logger.handlers = [handler]
    if verbose:
        environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf_logging.set_verbosity(tf_logging.INFO)
        logger.setLevel(logging.DEBUG)
    else:
        warnings.filterwarnings('ignore')
        environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf_logging.set_verbosity(tf_logging.ERROR)
