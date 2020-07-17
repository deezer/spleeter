#!/bin/bash

######################################################################
# Custom entrypoint that activate conda before running spleeter.
#
# @author Félix Voituret <fvoituret@deezer.com>
# @version 1.0.0
######################################################################

# shellcheck disable=1091
. "/opt/conda/etc/profile.d/conda.sh"
conda activate base
spleeter "$@"