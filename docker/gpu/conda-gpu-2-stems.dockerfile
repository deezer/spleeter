FROM researchdeezer/spleeter:conda-gpu

RUN mkdir -p /model/2stems \
    && wget -O /tmp/2stems.tar.gz https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz \
    && tar -xvzf /tmp/2stems.tar.gz -C /model/2stems/
