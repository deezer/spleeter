ARG BASE=python:3.7

FROM ${BASE} AS conda-installer

ARG CONDA_VERSION=3
ARG MINICONDA_VERSION=4.5.11

RUN apt-get update \
    && apt-get -y install wget \
    && wget --quiet \
        -O miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda${CONDA_VERSION}-${MINICONDA_VERSION}-Linux-x86_64.sh \
    && bash miniconda.sh -b -p /opt/conda

FROM ${BASE}

ENV PATH /opt/conda/bin:${PATH}
COPY --from=conda-installer /opt/conda/ /opt/conda/
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> "${HOME}/.bashrc" \
    && echo "conda activate base" >> "${HOME}/.bashrc"
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
