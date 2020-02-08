# spleeter

> Spleeter is the Deezer source separation library with pretrained models written in Python and uses Tensorflow.
> More information: <https://github.com/deezer/spleeter>.

- Vocals (singing voice) / accompaniment separation (2 stems):

`spleeter separate -i spleeter/audio_example.mp3 -p spleeter:2stems -o output`

- Vocals / drums / bass / other separation (4 stems)

`spleeter separate -i spleeter/audio_example.mp3 -p spleeter:4stems -o output`

- Vocals / drums / bass / piano / other separation (5 stems)

`spleeter separate -i spleeter/audio_example.mp3 -p spleeter:5stems -o output`
