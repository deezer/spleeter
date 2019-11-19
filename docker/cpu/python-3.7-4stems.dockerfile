FROM spleeter:3.7

RUN mkdir -p /model/4stems \
    && wget -O /tmp/4stems.tar.gz https://github.com/deezer/spleeter/releases/download/v1.4.0/4stems.tar.gz \
    && tar -xvzf /tmp/4stems.tar.gz -C /model/4stems/
