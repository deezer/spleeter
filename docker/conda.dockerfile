FROM continuumio/miniconda3:4.7.10

RUN mkdir -p /model
ENV MODEL_PATH /model
COPY audio_example.mp3 .

RUN conda install -y -c conda-forge musdb
# RUN conda install -y -c conda-forge museval
RUN conda install -y -c conda-forge spleeter=1.4.9

ENTRYPOINT ["spleeter"]