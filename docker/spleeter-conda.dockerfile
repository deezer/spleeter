ARG BASE=conda

FROM ${BASE}

ARG SPLEETER_VERSION=2.5.0
ENV MODEL_PATH /model

RUN mkdir -p /model
RUN conda config --add channels conda-forge
RUN conda install -y -c conda-forge musdb
RUN conda install -y -c deezer-research spleeter 
COPY docker/conda-entrypoint.sh spleeter-entrypoint.sh
ENTRYPOINT ["/bin/bash", "spleeter-entrypoint.sh"]