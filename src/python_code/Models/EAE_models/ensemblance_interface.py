from abc import abstractmethod
import tensorflow as tf
import numpy as np
import math


class EAE_interface(tf.keras.Model):

    def __init__(self, dim, **kwargs):
        """
        Wrapper for the EAE AutoEncoder (EAE)
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        :param num_labels: Number of labels for regularization
        :param dropout: Noise dropout [0,1]
        """
        self.h_dim = dim[0]
        self.z_dim = dim[1]
        self.image_size = dim[2]
        self.original_size = int(math.sqrt(self.image_size))
        super(EAE_interface, self).__init__(**kwargs)

    @staticmethod
    def apply_mask(m, weights):
        """
        Applying mask to a specific layer
        :param m:
        :param weights:
        :return:
        """
        weights[0] = tf.math.multiply(m, weights[0])
        return weights

    @staticmethod
    def get_mask(size, p=0.8):
        """
        Get a random mask
        :param size: shape of the mask
        :param p: probability of getting a 1 [1,0]
        :return:
        """
        aux = np.random.choice([1, 0], size, p=[p, 1 - p])
        while np.all(aux):
            aux = np.random.choice([1, 0], size, p=[p, 1 - p])
        return tf.convert_to_tensor(aux, dtype=tf.float32)

    @abstractmethod
    def initialize_masks(self):
        """
        Initialize masks for the model
        :return:
        """
        pass

    @abstractmethod
    def apply_masks(self):
        """
        Apply masks to all layers of the model
        :return:
        """
        pass

    @abstractmethod
    def encode(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        pass

    @abstractmethod
    def decode(self, z):
        """
        Decode without activation function
        :param z: latent space
        :return: output model
        """
        pass

    @abstractmethod
    def call(self, inputs, training=None, mask=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :return: model output
        """
        pass

    @abstractmethod
    def load_weights_model(self, list_path):
        """
        Load the weights of the model
        """
        pass

    @abstractmethod
    def save_weights_model(self, list_path):
        """
        Save the weights of the model
        """
        pass