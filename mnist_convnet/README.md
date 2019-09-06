# Convnet in emloop-tensorflow
This folder contains convolutional network example written in **emloop-tensorflow**
commented in greater detail in our [emloop-tensorflow tutorial](https://tensorflow.emloop.org/tutorial.html).

It is required to have **python 3.5+** and **pip** available in your system.

1. Install **emloop-tensorflow** and download the examples (if not done yet):
```
pip3 install emloop emloop-tensorflow --upgrade
git clone https://github.com/iterait/emloop-examples.git
cd emloop-examples
```

2. Download the data and train the network:
```
emloop dataset download mnist_convnet
emloop train mnist_convnet
```

3. Train on two GPUs (omit `model.n_gpus:int=2` in order to train on CPU):
```
emloop train mnist_convnet model.n_gpus:int=2
```
