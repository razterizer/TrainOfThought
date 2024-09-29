# TrainOfThought Lib

This is a header-only library.

![GitHub License](https://img.shields.io/github/license/razterizer/TrainOfThought?color=blue)

[![build and test ubuntu](https://github.com/razterizer/TrainOfThought/actions/workflows/build-and-test-ubuntu.yml/badge.svg)](https://github.com/razterizer/TrainOfThought/actions/workflows/build-and-test-ubuntu.yml)

![Top Languages](https://img.shields.io/github/languages/top/razterizer/TrainOfThought)
![Goto](https://img.shields.io/github/search/razterizer/TrainOfThought/goto)
![Commit Activity](https://img.shields.io/github/commit-activity/t/razterizer/TrainOfThought)
![Last Commit](https://img.shields.io/github/last-commit/razterizer/TrainOfThought?color=blue)
![Contributors](https://img.shields.io/github/contributors/razterizer/TrainOfThought?color=blue)
<!-- ![Languages](https://img.shields.io/github/languages/count/razterizer/TrainOfThought) -->

## Machine Learning

This part of the library will probably move out to its own repo one day.

The standard stuff work fairly well AFAIK, but I do have some problems with the LeNet-5 implementation (see below). 

The reason I wanted to implement LeNet-5 (by Yann LeCun) is to learn convolutional networks. My opinion is that you don't learn how these more advanced types of network work just by using `PyTorch` or `TensorFlow` etc, but you rather learn how to use those APIs. This is my attempt to actually understand how LeNet-5 works under the hood so that I can implement more advanced networks such as `AlexNet`. I've planned to add other types of machine learning techniques and networks such as `LSTM`s and `Q-learning` (RL) in the future.

## Note

The LeNet-5 code is a W.I.P. at the moment. The paper by Yann LeCun is a bit unclear on the details. The feed forward part is fairly straight forward, but issues start popping up when attempting to backprop the thing.
