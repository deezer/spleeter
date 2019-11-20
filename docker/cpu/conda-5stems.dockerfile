FROM researchdeezer/spleeter:conda

RUN mkdir -p /model/5stems \
    && wget -O /tmp/5stems.tar.gz https://github.com/deezer/spleeter/releases/download/v1.4.0/5stems.tar.gz \
    && tar -xvzf /tmp/5stems.tar.gz -C /model/5stems/
