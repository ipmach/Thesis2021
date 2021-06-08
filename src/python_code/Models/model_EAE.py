from Models.EAE_models.vanilla_ensemblance_cnn import Vanilla_CNN_ensemblance
from Models.EAE_models.vanilla_ensemblance import Vanilla_ensemblance
from Models.AAE_interface import AE
import tensorflow as tf


class RandNet(AE):

    def __init__(self,dim, p_m, num, cnn=False ,filters=None, **kwargs):
        super(RandNet, self).__init__(dim, **kwargs)
        self.num_models = num
        if cnn:
            self.preformat = lambda x: x
            self.postformat = lambda x: x
            self.__models__ = [Vanilla_CNN_ensemblance(filters, [self.h_dim, self.z_dim,
                                                        self.image_size], p_m)
                               for _ in range(self.num_models)]
        else:
            self.__models__ = [Vanilla_ensemblance([self.h_dim, self.z_dim,
                                                    self.image_size], p_m)
                           for _ in range(self.num_models)]
        self.concact_index = 1

    def build(self, input_shape):
        """
        Build of the models
        :param input_shape: input shape of the models
        :return:
        """
        for m in self.__models__:
            m.build(input_shape=input_shape)

    def initialize_masks(self):
        """
        Initialize all the different masks of all the models
        :return:
        """
        for m in self.__models__:
            m.initialize_masks()

    def apply_masks(self):
        """
        Apply all the mask of all the models
        :return:
        """
        for m in self.__models__:
            m.apply_masks()

    def encode(self, x, index=None):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        if index is None:
            return tf.convert_to_tensor([m.encode(x) for m in self.__models__],
                                        dtype=tf.float32)
        return self.__models__[index].encode(x)

    def decode(self, z, index=None):
        """
        Decode without activation function
        :param z: latent space
        :return: output model
        """
        if index is None:
            return tf.convert_to_tensor([m.decode(z) for m in self.__models__],
                                        dtype=tf.float32)
        return self.__models__[index].decode(z)

    def call_(self, inputs, training=None, mask=None, index=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :return: model output
        """
        if index is None:
            return tf.convert_to_tensor([m(inputs) for m in self.__models__],
                                        dtype=tf.float32)
        return self.__models__[index](inputs)

    def load_weights_model(self, list_path):
        """
        Load the weights of the model
        """
        for j, m in enumerate(self.__models__):
            m.load_weights_model(list_path[j])

    def save_weights_model(self, list_path):
        """
        Save the weights of the model
        """
        for j, m in enumerate(self.__models__):
            m.save_weights_model(list_path[j])

    def __getitem__(self, index):
        return self.__models__[index]

    def __len__(self):
        return len(self.__models__)