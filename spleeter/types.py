#!/usr/bin/env python
# coding: utf8

""" Custom types definition. """

from typing import Any, Tuple

# pyright: reportMissingImports=false
# pylint: disable=import-error
import numpy as np

# pylint: enable=import-error

AudioDescriptor = Any
Signal = Tuple[np.ndarray, float]
