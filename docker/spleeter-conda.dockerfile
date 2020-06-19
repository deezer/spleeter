ARG BASE=conda

FROM ${BASE}

ARG SPLEETER_PACKAGE=spleeter
ARG SPLEETER_VERSION=1.5.3
ENV MODEL_PATH /model

RUN mkdir -p /model
RUN conda install -y -c conda-forge musdb
RUN conda install -y -c conda-forge ${SPLEETER_PACKAGE}==${SPLEETER_VERSION}
COPY docker/conda-entrypoint.sh spleeter-entrypoint.sh
ENTRYPOINT ["/bin/bash", "spleeter-entrypoint.sh"]