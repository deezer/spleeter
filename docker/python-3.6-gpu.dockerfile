FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

# TODO: Install cuda !
RUN apt-get update && apt-get install -y ffmpeg libsndfile1
RUN pip install musdb museval
RUN pip install spleeter-gpu==1.4.4
RUN mkdir -p /model
ENV MODEL_PATH /model
ENTRYPOINT ["spleeter"]