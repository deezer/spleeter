#!/usr/bin/env python
# coding: utf8

""" Centralized logging facilities for Spleeter. """

from os import environ

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


class _LoggerHolder(object):
    """ Logger singleton instance holder. """

    INSTANCE = None


def get_logger():
    """ Returns library scoped logger.

    :returns: Library logger.
    """
    if _LoggerHolder.INSTANCE is None:
        # pylint: disable=import-error
        from tensorflow.compat.v1 import logging
        # pylint: enable=import-error
        _LoggerHolder.INSTANCE = logging
        _LoggerHolder.INSTANCE.set_verbosity(_LoggerHolder.INSTANCE.ERROR)
        environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    return _LoggerHolder.INSTANCE


def enable_logging():
    """ Enable INFO level logging. """
    environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    logger = get_logger()
    logger.set_verbosity(logger.INFO)


def enable_verbose_logging():
    """ Enable DEBUG level logging. """
    environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logger = get_logger()
    logger.set_verbosity(logger.DEBUG)
