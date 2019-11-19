FROM continuumio/miniconda3:4.7.10

RUN conda install -y -c conda-forge spleeter
RUN mkdir -p /model
ENV MODEL_PATH /model

ENTRYPOINT ["spleeter"]