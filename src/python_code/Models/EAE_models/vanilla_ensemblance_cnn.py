from Models.EAE_models.ensemblance_interface import EAE_interface
from Models.EAE_models.EncoderCNN import EncoderCNN
from Models.EAE_models.DecoderCNN import DecoderCNN
import tensorflow as tf


class Vanilla_CNN_ensemblance(EAE_interface):

    def __init__(self, filters, dim, p_m, **kwargs):
        super(Vanilla_CNN_ensemblance, self).__init__(dim, **kwargs)
        self.fc1 = tf.keras.layers.Dense(self.z_dim)
        self.fc2 = tf.keras.layers.Dense(4 * 4 * 32)
        self.encoder_ = EncoderCNN(filters, p_m, self.get_mask, self.apply_mask)
        self.decoder_ = DecoderCNN(filters, p_m, self.get_mask, self.apply_mask)
        self._layers_ = [self.fc1, self.fc2, self.decoder_]
        self._layers_cnn = [self.encoder_, self.decoder_]
        self._masks_ = None
        self.p_m = p_m

    def initialize_masks(self):
        """
        Initialize masks for the model
        :return:
        """
        self._masks_ = [self.get_mask(i.get_weights()[0].shape,
                                      p=self.p_m)
                        for i in self._layers_]
        for i in self._layers_cnn:
          i.initialize_masks()

    def apply_masks(self):
        """
        Apply masks to all layers of the model
        :return:
        """
        for l, m in zip(self._layers_, self._masks_):
            new_weights = self.apply_mask(m, l.get_weights())
            l.set_weights(new_weights)

        for i in self._layers_cnn:
          i.apply_masks()

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

    def call(self, inputs, training=None, mask=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :return: model output
        """
        z = self.encode(inputs)
        return self.decode(z)

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