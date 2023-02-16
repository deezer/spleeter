#!/usr/bin/env python
# coding: utf8

"""
Spleeter is the Deezer source separation library with pretrained models.
The library is based on Tensorflow:

-   It provides already trained model for performing separation.
-   It makes it easy to train source separation model with tensorflow
    (provided you have a dataset of isolated sources).

This module allows to interact easily from command line with Spleeter
by providing train, evaluation and source separation action.
"""

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


class SpleeterError(Exception):
    """Custom exception for Spleeter related error."""

    pass
