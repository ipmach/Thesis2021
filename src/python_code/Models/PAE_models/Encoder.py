import tensorflow as tf


class Encoder(tf.keras.Model):

    def __init__(self, dim, **kwargs):
        """
        Encoder model
        :param dim: hyperparameters of the model [h_dim, z_dim]
        :param dropout: Noise dropout [0,1]
        :param kwargs: Keras parameters (Optional)
        """
        h_dim = dim[0]
        z_dim = dim[1]
        super(Encoder, self).__init__(**kwargs)

        self.fc1 = tf.keras.layers.Dense(h_dim)
        self.fc2 = tf.keras.layers.Dense(z_dim)

    def call(self, inputs, training=None, mask=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :return: model output
        """
        h = tf.nn.relu(self.fc1(inputs))
        z = self.fc2(h)
        return z