ARG BASE=conda
ARG SPLEETER_PACKAGE=spleeter
ARG SPLEETER_VERSION=1.5.3

FROM ${BASE}

ENV MODEL_PATH /model

RUN mkdir -p /model
COPY audio_example.mp3 .

RUN conda install -y -c conda-forge musdb
RUN conda install -y -c conda-forge ${SPLEETER_PACKAGE}==${SPLEETER_VERSION}

ENTRYPOINT ["spleeter"]