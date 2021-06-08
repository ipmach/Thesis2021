from Models.CNN_layers.Decoder_CNN import DecoderCNN
from Models.AE_CNN_interface import AE_CNN
import tensorflow as tf


class VAECNN(AE_CNN):

    def __init__(self, filters, z_dim):
        super(VAECNN, self).__init__(filters, z_dim)
        self.fc1 = tf.keras.layers.Dense(self.z_dim)
        self.fc3 = tf.keras.layers.Dense(z_dim)
        self.fc2 = tf.keras.layers.Dense(4 * 4 * 32)
        self.decoder_ = DecoderCNN(filters, last_activation=None)
        input_shape = (self.img_shape[0], self.img_shape[1], self.img_shape[2])
        self.encode_ = tf.keras.Sequential([tf.keras.Input(shape=input_shape),
                                            self.encoder_, self.fc1])

    def encode(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        x = self.encoder_(x)
        z = self.fc1(x)
        return z

    def encode_both(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        x = self.encoder_(x)
        mu = self.fc1(x)
        log_var = self.fc3(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization of z
        :param mu: dense 1
        :param log_var: dense 2
        :return: mu + eps * std
        """
        std = tf.exp(log_var * 0.5)
        eps = tf.random.normal(tf.shape(std))
        return mu + eps * std

    def decode(self, z):
        """
        Decode without activation function
        :param z: latent space
        :return: output model
        """
        x = self.fc2(z)
        x = self.decoder_(x)
        return x

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
            encoded = self.reparameterize(mu, log_var)
            reconstructed = self.decode(encoded)
            return reconstructed, mu, log_var
        else:
            mu, _ = self.encode_both(inputs)
            reconstructed = self.decode(mu)
            return tf.math.sigmoid(reconstructed)

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