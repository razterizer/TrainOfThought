# Core Lib

This is a header-only library.

![GitHub License](https://img.shields.io/github/license/razterizer/Core?color=blue)


### DateTime.h

The code in this header file is put inside a namespace `datetime` which contain classes `Date`, `Time` and `DateTime`.

### Delay.h

The code here is in namespace `Delay` and features two functions `sleep(T us)` and `update_loop(int fps, std::function<bool(void)> update_func)`.

## Machine Learning

This part of the library will probably move out to its own repo one day.

The standard stuff work fairly well AFAIK, but I do have some problems with the LeNet-5 implementation (see below). 

The reason I wanted to implement LeNet-5 (by Yann LeCun) is to learn convolutional networks. My opinion is that you don't learn how these more advanced types of network work just by using `PyTorch` or `TensorFlow` etc, but you rather learn how to use those APIs. This is my attempt to actually understand how LeNet-5 works under the hood so that I can implement more advanced networks such as `AlexNet`. I've planned to add other types of machine learning techniques and networks such as `LSTM`s and `Q-learning` (RL) in the future.

## Note

The LeNet-5 code is a W.I.P. at the moment. The paper by Yann LeCun is a bit unclear on the details. The feed forward part is fairly straight forward, but issues start popping up when attempting to backprop the thing.
