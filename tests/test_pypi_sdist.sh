#!/bin/bash

######################################################################
# Script that performs PyPi packaging test.
#
# @author Deezer Research <research@deezer.com>
# @version 1.0.0
######################################################################

	twine upload --repository-url https://test.pypi.org/legacy/ dist/*