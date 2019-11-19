FROM continuumio/miniconda3:4.7.10

# install tensorflow
RUN conda install -y tensorflow==1.14.0

# install ffmpeg for audio loading/writing
RUN conda  install -y -c conda-forge ffmpeg

# install extra python libraries
RUN conda install -y -c anaconda pandas==0.25.1
RUN conda install -y -c conda-forge libsndfile

# install ipython
RUN conda install -y ipython

WORKDIR /workspace/
COPY ./ spleeter/

RUN mkdir /cache/

WORKDIR /workspace/spleeter
RUN pip install .

ENTRYPOINT ["python", "-m", "spleeter"]