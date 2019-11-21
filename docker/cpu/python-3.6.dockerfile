FROM python:3.6

RUN apt-get update && apt-get install -y ffmpeg libsndfile
RUN pip install musdb museval
RUN pip install spleeter
RUN mkdir -p /model
ENV MODEL_PATH /model
ENTRYPOINT ["spleeter"]