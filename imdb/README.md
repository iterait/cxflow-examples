# IMDB Review Sentiment Dataset
In this example, we train a simple GRU recurrent network for sentiment analysis for movie reviews.

The network is trained on IMDB dataset <http://ai.stanford.edu/~amaas/data/sentiment/>.

Trained network can be easily applied to a new review as explained bellow.

It is required to have **python 3.5+** and **pip** available in your system.

**Compatible with TensorFlow version 1.4.**

For a different version checkout appropriate branch after clonning, e.g.:
```
git checkout tf-1.3
```

1. Install **emloop-tensorflow** and download the examples (if not done yet):
```
pip3 install emloop emloop-tensorflow --upgrade
git clone https://github.com/iterait/emloop-examples.git
cd emloop-examples
```

2. Train the bi-directional GRU network
```
emloop train imdb
```

You should reach test accuracy around 89.5%, the best model will be saved in `log/IMDB_GRU_<dir_name>`.

3. Obtain predictions interactively:
```
emloop predict log/<your training directory>
```

After a moment, you should see the following prompt:
```
Type the review or leave empty to end: I was not sure at the beginning, but in the end it was a good movie. The performance of Emma Stone was outstanding.
Predicted sentiment: positive

Type the review or leave empty to end: I like Tarantino films, but this one was simply awful. The dialogues did not make any sense and the camera was terrible and shaky.
Predicted sentiment: negative

Type the review or leave empty to end:
```

Can you fool your network?
