FROM continuumio/miniconda3:4.7.10

RUN conda install -y ipython \
    && conda install -y tensorflow-gpu==1.14.0 \
    && conda install -y -c conda-forge ffmpeg \
    && conda install -y -c conda-forge libsndfile \
    && conda install -y -c anaconda pandas==0.25.1 \
RUN mkdir -p /model
ENV MODEL_PATH /model
RUN pip install spleeter

ENTRYPOINT ["spleeter"]