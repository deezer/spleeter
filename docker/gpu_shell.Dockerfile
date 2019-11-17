FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

# set work directory
WORKDIR /workspace

# install anaconda
ENV PATH /opt/conda/bin:$PATH
COPY install_miniconda.sh .
RUN bash ./install_miniconda.sh && rm install_miniconda.sh

RUN conda update -n base -c defaults conda

# install tensorflow for GPU
RUN conda install -y tensorflow-gpu==1.14.0

# install ffmpeg for audio loading/writing
RUN conda  install -y -c conda-forge ffmpeg

# install extra libs
RUN conda install -y -c anaconda pandas==0.25.1
RUN conda install -y -c conda-forge libsndfile

# install ipython
RUN conda install -y ipython

RUN mkdir /cache/

# clone inside image github repository
COPY ./ spleeter/

WORKDIR /workspace/spleeter

RUN apt-get update && apt-get install -y sudo
COPY with_the_same_user.sh /with_the_same_user.sh

RUN echo "export PATH=\$HOME/.local/bin:\$PATH" >>/etc/profile
RUN echo "pip install --user ." >>/etc/profile
