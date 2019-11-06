#!/usr/bin/env python
# coding: utf8

""" Centralized logging facilities for Spleeter. """

import logging

from os import environ

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

_FORMAT = '%(levelname)s:%(name)s:%(message)s'


class _LoggerHolder(object):
    """ Logger singleton instance holder. """

    INSTANCE = None


def get_tensorflow_logger():
    """
    """
    # pylint: disable=import-error
    from tensorflow.compat.v1 import logging
    # pylint: enable=import-error
    return logging


def get_logger():
    """ Returns library scoped logger.

    :returns: Library logger.
    """
    if _LoggerHolder.INSTANCE is None:
        formatter = logging.Formatter(_FORMAT)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger('spleeter')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        _LoggerHolder.INSTANCE = logger
    return _LoggerHolder.INSTANCE


def enable_tensorflow_logging():
    """ Enable tensorflow logging. """
    environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf_logger = get_tensorflow_logger()
    tf_logger.set_verbosity(tf_logger.INFO)
    logger = get_logger()
    logger.setLevel(logging.DEBUG)


def enable_logging():
    """ Configure default logging. """
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf_logger = get_tensorflow_logger()
    tf_logger.set_verbosity(tf_logger.ERROR)
