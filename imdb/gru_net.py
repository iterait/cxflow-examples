import emloop_tensorflow as eltf
import tensorflow as tf


class SimpleGRU(eltf.BaseModel):
    """Simple 2-layered MLP for majority task."""

    def _create_model(self, dropout: int=0.5, embedding_dim: int=256, gru_dim: int=128):
        x = tf.placeholder(dtype=tf.float32, shape=[None, self._dataset.maxlen], name='x')
        y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')

        with tf.variable_scope('emb1'):
            net = tf.keras.layers.Embedding(self._dataset.vocab_size, embedding_dim)(x)
            net = tf.keras.layers.Dropout(0.1)(net, training=self.is_training)
        with tf.variable_scope('gru2'):
            net = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_dim, return_sequences=True))(net)
            net = tf.reduce_mean(net, axis=1)
        with tf.variable_scope('dense4'):
            net = tf.keras.layers.Dropout(dropout)(net,  training=self.is_training)
            logits = tf.keras.layers.Dense(2, activation=None)(net)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        tf.identity(loss, name='loss')
        predictions = tf.argmax(logits, 1, name='predictions')
        tf.equal(predictions, y, name='accuracy')
