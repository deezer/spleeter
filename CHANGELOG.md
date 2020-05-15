# Changelog History

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