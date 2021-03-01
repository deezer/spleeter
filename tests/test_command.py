#!/usr/bin/env python
# coding: utf8

""" Unit testing for Separator class. """

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

from spleeter.__main__ import spleeter
from typer.testing import CliRunner


def test_version():

    runner = CliRunner()

    # execute spleeter version command
    result = runner.invoke(spleeter, [
        '--version',
    ])