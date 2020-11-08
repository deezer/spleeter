ARG BASE=researchdeezer/spleeter

FROM ${BASE}

ARG MODEL=2stems
RUN mkdir -p /model/$MODEL \
    && wget -O /tmp/$MODEL.tar.gz https://github.com/deezer/spleeter/releases/download/v1.4.0/$MODEL.tar.gz \
    && tar -xvzf /tmp/$MODEL.tar.gz -C /model/$MODEL/ \
    && touch /model/$MODEL/.probe
