# Majority example
This is the very first toy example for **emloop** framework commented in greater detail in our [tutorial](https://emloop.org/tutorial.html).

It is required to have **python 3.5+** and **pip** available in your system.

1. Install **emloop-tensorflow** and download the examples (if not done yet):
```
pip3 install emloop emloop-tensorflow --upgrade
git clone https://github.com/iterait/emloop-examples.git
cd emloop-examples
```

2. Train the majority network
```
emloop train majority
```

The best network will be saved in `log/MajorityExample_<dir_name>`.

3. Resume the training
```
emloop resume log/MajorityExample_<dir_name>
```

The very first epoch should perform similarly well to the last epoch of the previous training.
