from Models.AE_CNN_interface import AE_CNN
import tensorflow as tf


class DAECNN(AE_CNN):

    def __init__(self, filters, z_dim):
        super(DAECNN, self).__init__(filters, z_dim)
        self.fc1 = tf.keras.layers.Dense(self.z_dim)
        self.fc2 = tf.keras.layers.Dense(4 * 4 * 32)

    def encode(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        x = self.encoder_(x)
        z = self.fc1(x)
        return z

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
        encoded = self.encode(inputs)
        reconstructed = self.decode(encoded)
        return reconstructed

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