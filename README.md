<img src="https://github.com/deezer/spleeter/raw/master/images/spleeter_logo.png" height="80" />

[![Github actions](https://github.com/deezer/spleeter/workflows/pytest/badge.svg)](https://github.com/deezer/spleeter/actions) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spleeter) [![PyPI version](https://badge.fury.io/py/spleeter.svg)](https://badge.fury.io/py/spleeter) [![Conda](https://img.shields.io/conda/vn/conda-forge/spleeter)](https://anaconda.org/conda-forge/spleeter) [![Docker Pulls](https://img.shields.io/docker/pulls/researchdeezer/spleeter)](https://hub.docker.com/r/researchdeezer/spleeter) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deezer/spleeter/blob/master/spleeter.ipynb) [![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/spleeter/community) [![status](https://joss.theoj.org/papers/259e5efe669945a343bad6eccb89018b/status.svg)](https://joss.theoj.org/papers/259e5efe669945a343bad6eccb89018b)

## About

**Spleeter** is a source separation library written in [Python](https://www.python.org/) and utilizing [Tensorflow](https://tensorflow.org/), created by [Deezer](https://www.deezer.com/). It makes it easy to train source separation models (assuming you have a dataset of isolated sources), and provides already-trained state-of-the-art models for performing various flavours of separation:

* Vocals (singing voice) / accompaniment separation ([2 stems](https://github.com/deezer/spleeter/wiki/2.-Getting-started#using-2stems-model))
* Vocals / drums / bass / other separation ([4 stems](https://github.com/deezer/spleeter/wiki/2.-Getting-started#using-4stems-model))
* Vocals / drums / bass / piano / other separation ([5 stems](https://github.com/deezer/spleeter/wiki/2.-Getting-started#using-5stems-model))

2 stem and 4 stem models have [high performances](https://github.com/deezer/spleeter/wiki/Separation-Performances) on the [musdb](https://sigsep.github.io/datasets/musdb.html) dataset. **Spleeter** is also very fast - it can perform separation of audio files to 4 stems **100x faster than real-time** when run on a GPU.

We designed **Spleeter** so you can use it straight from the [command line](https://github.com/deezer/spleeter/wiki/2.-Getting-started#usage),
as well as directly in your own development pipeline as a [Python library](https://github.com/deezer/spleeter/wiki/4.-API-Reference#separator). It can be installed with [Conda](https://github.com/deezer/spleeter/wiki/1.-Installation#using-conda),
with [pip](https://github.com/deezer/spleeter/wiki/1.-Installation#using-pip) or used with
[Docker](https://github.com/deezer/spleeter/wiki/2.-Getting-started#using-docker-image).


## Quick start

Want to try out Spleeter but don't want to install anything? We have a [Google Colab](https://colab.research.google.com/github/deezer/spleeter/blob/master/spleeter.ipynb) set up. Simply click the play button next to each command to run it on the Google Colab system. This will allow you to try separating the vocals and accompaniment stems from an example audio file.

Ready to dig into it? In a few lines you can install **Spleeter** using [Conda](https://github.com/deezer/spleeter/wiki/1.-Installation#using-conda) and separate the vocal and accompaniment parts from an example audio file:

```bash
# install using conda
conda install -c conda-forge spleeter
# download an example audio file (if you don't have wget, use another tool for downloading)
wget https://github.com/deezer/spleeter/raw/master/audio_example.mp3
# separate the example audio into two components
spleeter separate -i audio_example.mp3 -p spleeter:2stems -o output
```

You should get two separated audio files (`vocals.wav` and `accompaniment.wav`) in the `output/audio_example` folder.

For detailed documentation, please check the [repository wiki](https://github.com/deezer/spleeter/wiki)

## Development and Testing

The following set of commands will clone this repository, create a virtual environment provisioned with the dependencies and run the tests (Which will take a few minutes):

```bash
git clone https://github.com/Deezer/spleeter && cd spleeter
python -m venv spleeterenv && source spleeterenv/bin/activate
pip install . && pip install pytest pytest-xdist
make test
```

## Reference

* Deezer Research - Source Separation Engine Story - deezer.io blog post:
  * [English version](https://deezer.io/releasing-spleeter-deezer-r-d-source-separation-engine-2b88985e797e)
  * [Japanese version](http://dzr.fm/splitterjp)
* [Music Source Separation tool with pre-trained models / ISMIR2019 extended abstract](http://archives.ismir.net/ismir2019/latebreaking/000036.pdf)

If you use **Spleeter** in your work, please cite:

```BibTeX
@article{spleeter2020,
  doi = {10.21105/joss.02154},
  url = {https://doi.org/10.21105/joss.02154},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {50},
  pages = {2154},
  author = {Romain Hennequin and Anis Khlif and Felix Voituret and Manuel Moussallam},
  title = {Spleeter: a fast and efficient music source separation tool with pre-trained models},
  journal = {Journal of Open Source Software},
  note = {Deezer Research}
}
```

## License

The code of **Spleeter** is [MIT-licensed](LICENSE).

## Disclaimer

If you plan to use Spleeter on copyrighted material, make sure you get proper authorization from right owners beforehand.

### Forks and related projects

As is commonly the case with open-source projects, there are multiple forks exposing **Spleeter** through either a Guided User Interface (GUI) or a standalone free or paying website. Please note that we do not host, maintain or directly support any of these initiatives.

## Troubleshooting

**Spleeter** is a complex piece of software and although we continously try to improve and test it, you may encounter unexpected issues running it. If that's the case, please check the [FAQ page](https://github.com/deezer/spleeter/wiki/5.-FAQ) first, as well as the list of [currently open issues](https://github.com/deezer/spleeter/issues).

### Windows users

   It appears that sometimes the shortcut command `spleeter` does not work properly on windows. This is a known issue that we will hopefully fix soon. In the meantime replace `spleeter separate` with `python -m spleeter separate` in the command line and it should work.

## Contributing

If you would like to participate in the development of **Spleeter** you are more than welcome to do so. Don't hesitate to throw us a pull request and we'll do our best to examine it quickly. Please check out our [contributing guidelines](.github/CONTRIBUTING.md) first.

## Note

This repository includes a demo audio file `audio_example.mp3` which is an excerpt
from Slow Motion Dream by Steven M Bryant (c) copyright 2011 Licensed under a Creative
Commons Attribution (3.0) [license](http://dig.ccmixter.org/files/stevieb357/34740)
Ft: CSoul,Alex Beroza & Robert Siekawitch
