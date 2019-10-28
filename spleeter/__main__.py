#!/usr/bin/env python
# coding: utf8

"""
    Python oneliner script usage.

    USAGE: python -m spleeter {train,evaluate,separate} ...
"""

import sys
import warnings

from .commands import create_argument_parser
from .utils.configuration import load_configuration
from .utils.logging import enable_logging, enable_verbose_logging

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


def main(argv):
    """ Spleeter runner. Parse provided command line arguments
    and run entrypoint for required command (either train,
    evaluate or separate).

    :param argv: Provided command line arguments.
    """
    parser = create_argument_parser()
    arguments = parser.parse_args(argv[1:])
    if arguments.verbose:
        enable_verbose_logging()
    else:
        enable_logging()
    if arguments.command == 'separate':
        from .commands.separate import entrypoint
    elif arguments.command == 'train':
        from .commands.train import entrypoint
    elif arguments.command == 'evaluate':
        from .commands.evaluate import entrypoint
    params = load_configuration(arguments.params_filename)
    entrypoint(arguments, params)


def entrypoint():
    """ Command line entrypoint. """
    warnings.filterwarnings('ignore')
    main(sys.argv)


if __name__ == '__main__':
    entrypoint()
