<img src="https://github.com/deezer/spleeter/raw/master/images/spleeter_logo.png" height="80" />

[![PyPI version](https://badge.fury.io/py/spleeter.svg)](https://badge.fury.io/py/spleeter) ![Conda](https://img.shields.io/conda/dn/conda-forge/spleeter)

## About

**Spleeter** is the [Deezer](https://www.deezer.com/) source separation library with pretrained models
written in [Python](https://www.python.org/) and uses [Tensorflow](tensorflow.org/). It makes it easy
to train source separation model (assuming you have a dataset of isolated sources), and provides
already trained state of the art model for performing various flavour of separation :

* Vocals (singing voice) / accompaniment separation ([2 stems](https://github.com/deezer/spleeter/wiki/2.-Getting-started#using-2stems-model))
* Vocals / drums / bass / other separation ([4 stems](https://github.com/deezer/spleeter/wiki/2.-Getting-started#using-4stems-model))
* Vocals / drums / bass / piano / other separation ([5 stems](https://github.com/deezer/spleeter/wiki/2.-Getting-started#using-5stems-model))

2 stems and 4 stems models have state of the art performances on the [musdb](https://sigsep.github.io/datasets/musdb.html) dataset. **Spleeter** is also very fast as it can perform separation of audio files to 4 stems 100x faster than real-time when run on a GPU. 

We designed **Spleeter** so you can use it straight from [command line](https://github.com/deezer/spleeter/wiki/2.-Getting-started#usage)
as well as directly in your own development pipeline as a [Python library](https://github.com/deezer/spleeter/wiki/4.-API-Reference#separator). It can be installed with [Conda](https://github.com/deezer/spleeter/wiki/1.-Installation#using-conda),
with [pip](https://github.com/deezer/spleeter/wiki/1.-Installation#using-pip) or be used with
[Docker](https://github.com/deezer/spleeter/wiki/2.-Getting-started#using-docker-image).

## Quick start 

Want to try it out ? Just clone the repository and install a
[Conda](https://github.com/deezer/spleeter/wiki/1.-Installation#using-conda)
environment to start separating audio file as follows:

```bash
$ git clone https://github.com/Deezer/spleeter
$ conda env create -f spleeter/conda/spleeter-cpu.yaml
$ conda activate spleeter-cpu
$ spleeter separate -i spleeter/audio_example.mp3 -p spleeter:2stems -o output
```
You should get two separated audio files (`vocals.wav` and `accompaniment.wav`)
in the `output/audio_example` folder.

For a more detailed documentation, please check the [repository wiki](https://github.com/deezer/spleeter/wiki)

## Reference
If you use **Spleeter** in your work, please cite:

```
@misc{spleeter2019,
  title={Spleeter: A Fast And State-of-the Art Music Source Separation Tool With Pre-trained Models},
  author={Romain Hennequin and Anis Khlif and Felix Voituret and Manuel Moussallam},
  howpublished={Late-Breaking/Demo ISMIR 2019},
  month={November},
  year={2019}
}
```

## License
The code of **Spleeter** is MIT-licensed.

## Note
This repository include a demo audio file `audio_example.mp3` which is an excerpt
from Slow Motion Dream by Steven M Bryant (c) copyright 2011 Licensed under a Creative
Commons Attribution (3.0) license. http://dig.ccmixter.org/files/stevieb357/34740
Ft: CSoul,Alex Beroza & Robert Siekawitch
