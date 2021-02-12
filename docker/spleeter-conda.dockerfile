ARG BASE=conda

FROM ${BASE}

ARG SPLEETER_VERSION=1.5.3
ENV MODEL_PATH /model

RUN mkdir -p /model
RUN conda install -y -c conda-forge musdb
RUN conda install -y -c conda-forge spleeter==${SPLEETER_VERSION}
COPY docker/conda-entrypoint.sh spleeter-entrypoint.sh
ENTRYPOINT ["/bin/bash", "spleeter-entrypoint.sh"]