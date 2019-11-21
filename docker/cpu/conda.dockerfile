FROM continuumio/miniconda3:4.7.10

RUN conda install -y -c conda-forge musdb
# RUN conda install -y -c conda-forge museval
RUN conda install -y -c conda-forge spleeter=1.4.4
RUN mkdir -p /model
ENV MODEL_PATH /model

ENTRYPOINT ["spleeter"]