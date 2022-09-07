# Changelog History

## 2.3.2

Release contrain on specific Tensorflow, numpy and Librosa versions
Dropping explicit support of python 3.6 but adding 3.10
## 2.3.0

Updating dependencies to enable TensorFlow 2.5 support (and Python 3.9 overall)
Removing the destructor from the `Separator` class

## 2.2.0

Minor changes mainly fixing some issues:
* mono training was not working due to hardcoded filters in the dataset
* default argument of `separate` was of wrong type
* added a way to request spleeter version with the `--version` argument in the CLI

## 2.1.0

This version introduce design related changes, especially transition to Typer for CLI managment and Poetry as
library build backend.

* `-i` option is now deprecated and replaced by traditional CLI input argument listing
* Project is now built using Poetry
* Project requires code formatting using Black and iSort
* Dedicated GPU package `spleeter-gpu` is not supported anymore, `spleeter` package will support both CPU and GPU hardware

### API changes:

* function `get_default_audio_adapter` is now available as `default()` class method within `AudioAdapter` class
* function `get_default_model_provider` is now available as `default()` class method within `ModelProvider` class
* `STFTBackend` and `Codec` are now string enum
* `GithubModelProvider` now use `httpx` with HTTP/2 support
* Commands are now located in `__main__` module, wrapped as simple function using Typer options module provide specification for each available option and argument
* `types` module provide custom type specification and must be enhanced in future release to provide more robust typing support with MyPy
* `utils.logging` module has been cleaned, logger instance is now a module singleton, and a single function is used to configure it with verbose parameter
* Added a custom logger handler (see tiangolo/typer#203 discussion)


## 2.0

First release, October 9th 2020

Tensorflow-2 compatible version, allowing uses in python 3.8.

## 1.5.4

First release, July 24th 2020

Add some padding of the input waveform to avoid separation artefacts on the edges due to unstabilities in the inverse fourier transforms.
Also add tests to ensure both librosa and tensorflow backends have same outputs.

## 1.5.2

First released, May 15th 2020

### Major changes

* PR #375 merged to avoid mutliple tf.graph instantiation failures

### Minor changes

* PR #362 use tf.abs instead of numpy
* PR #352 tempdir cleaning


## 1.5.1

First released, April 15th 2020

### Major changes

* Bugfixes on the LibRosa STFT backend

### Minor changes

* Typos, and small bugfixes

## 1.5.0

First released, March 20th 2020

### Major changes

* Implement a new STFT backend using LibRosa, faster on CPU than TF implementation
* Switch tensorflow version to 1.15.2

### Minor changes

* Typos, and small bugfixes

## 1.4.9

First released, Dec 27th 2019

### Major changes

* Add new configuration for processing until 16Khz

### Minor changes

* Typos, and small bugfixes
