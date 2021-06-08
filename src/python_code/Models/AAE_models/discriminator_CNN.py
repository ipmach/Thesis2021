import tensorflow as tf


class Discriminator(tf.keras.Model):

    def __init__(self, dim, **kwargs):
        """
        Discriminator model
        :param dim: hyperparameters of the model [h_dim]
        :param dropout: Noise dropout [0,1]
        :param kwargs: Keras parameters (Optional)
        """
        h_dim = dim[0]
        super(Discriminator, self).__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(h_dim)
        self.fc2 = tf.keras.layers.Dense(h_dim)
        self.fc3 = tf.keras.layers.Dense(1)
        self.lru1 = tf.keras.layers.LeakyReLU()
        self.lru2 = tf.keras.layers.LeakyReLU()


    def call(self, inputs, training=None, mask=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :return: model output
        """
        h = self.lru1(self.fc1(inputs))
        h = self.lru2(self.fc2(h))
        x = self.fc3(h)
        return tf.reshape(x, (-1, 1))