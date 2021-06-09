import tensorflow as tf
from Models.AAE_interface import AE


class VAE(AE):

    def __init__(self, dim, **kwargs):
        """
        Wrapper for the Varational AutoEncoder (VAE)
        Inspired from: Deep Learning with Tensorflow 2 and Keras book
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        :param num_labels: Number of labels for regularization
        :param dropout: Noise dropout [0,1]
        """
        super(VAE, self).__init__(dim, **kwargs)

        self.fc1 = tf.keras.layers.Dense(self.h_dim, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(self.z_dim)
        self.fc3 = tf.keras.layers.Dense(self.z_dim)

        self.fc4 = tf.keras.layers.Dense(self.h_dim)
        self.fc5 = tf.keras.layers.Dense(self.image_size)
        self.encode_ = tf.keras.Sequential([tf.keras.Input(shape=(self.image_size)),
                                            self.fc1, self.fc2])

    def encode_both(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        h = tf.nn.relu(self.fc1(x))
        #h = self.fc1(x)
        return self.fc2(h), self.fc3(h)

    def encode(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        h = tf.nn.relu(self.fc1(x))
        return self.fc2(h)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization of z
        :param mu: dense 1
        :param log_var: dense 2
        :return: mu + eps * std
        """
        std = tf.exp(log_var * 0.5)
        eps = tf.random.normal(std.shape)
        return mu + eps * std

    def decode_logits(self, z):
        """
        Decode without activation function
        :param z: latent space
        :return: output model
        """
        h = tf.nn.relu(self.fc4(z))
        return self.fc5(h)

    def decode(self, z):
        """
        Decode with activation function sigmoid
        :param z: latent space
        :return: output model
        """
        return tf.nn.sigmoid(self.decode_logits(z))

    def call_(self, inputs, training=None, mask=None, index=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :return: model output
        """
        if training:
            mu, log_var = self.encode_both(inputs)
            z = self.reparameterize(mu, log_var)
            x_reconstructed_logits = self.decode_logits(z)
            return x_reconstructed_logits, mu, log_var
        else:
            return self.decode(self.encode(inputs))

    def load_weights_model(self, list_path):
        """
        Load the weights of the model
        """
        self.load_weights(list_path[0])

    def save_weights_model(self, list_path):
        """
        Save the weights of the model
        """
        self.save_weights(list_path[0])