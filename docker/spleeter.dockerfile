ARG BASE=python:3.12

FROM ${BASE}

ARG SPLEETER_VERSION=2.5.0
ENV MODEL_PATH /model

RUN mkdir -p /model
RUN apt-get update && apt-get install -y ffmpeg libsndfile1
RUN pip install musdb museval
RUN pip install spleeter==${SPLEETER_VERSION}

ENTRYPOINT ["spleeter"]
