from abc import abstractmethod
import tensorflow as tf
import numpy as np
import math
import json


class AE(tf.keras.Model):

    def __init__(self, dim, **kwargs):
        """
        Autoencoder Interface
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        """
        self.h_dim = dim[0]
        self.z_dim = dim[1]
        self.image_size = dim[2]
        self.original_size = int(math.sqrt(self.image_size))
        super(AE, self).__init__(**kwargs)
        self.preprocessing = lambda x: x / 255.
        self.postprocessing = lambda x: x * 255
        self.preformat = lambda x: x.reshape((len(x),
                                              np.prod(x.shape[1:])))
        """
        self.postformat = lambda x: x.reshape((len(x),
                                               self.original_size,
                                               self.original_size))
        """
        path_json = 'src/python_code/settings.json'
        settings = json.load(open(path_json))["Model"]
        size = settings["sizes"][int(settings["size_use"])]
        self.img_shape = (int(size[0]), int(size[1]), int(size[2]))
        self.postformat = lambda x: x.reshape((len(x), int(size[0]),
                                               int(size[1]), int(size[2])))
        self.concact_index = 0  # Index to concat batch size

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
        Decode with activation function sigmoid
        :param z: latent space
        :return: output model
        """
        pass

    @abstractmethod
    def call_(self, inputs, training=None, mask=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :return
        """
        pass

    def call(self, inputs, training=False, mask=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :use_batch: use batch or not
        :batch_size: size of the batch
        :return
        """
        outputs = self.call_(inputs, training=training, mask=mask)
        return outputs

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