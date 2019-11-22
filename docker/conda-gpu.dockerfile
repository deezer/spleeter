FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update --fix-missing \
    && apt-get install -y wget bzip2 ca-certificates curl git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda clean -tipsy \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && ln -s /opt/conda/bin/conda /usr/bin/conda

COPY audio_example.mp3 .

RUN conda install -y -c conda-forge musdb
# RUN conda install -y -c conda-forge museval
RUN conda install -y -c conda-forge spleeter-gpu=1.4.5


ENTRYPOINT ["spleeter"]