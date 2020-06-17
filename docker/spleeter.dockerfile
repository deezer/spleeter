ARG BASE=python:3.6
ARG SPLEETER_PACKAGE=spleeter
ARG SPLEETER_VERSION=1.5.3

FROM ${BASE}

ENV MODEL_PATH /model

RUN mkdir -p /model
COPY audio_example.mp3 .

RUN apt-get update && apt-get install -y ffmpeg libsndfile1
RUN pip install musdb museval
RUN pip install ${SPLEETER_PACKAGE}==${SPLEETER_VERSION}

ENTRYPOINT ["spleeter"]