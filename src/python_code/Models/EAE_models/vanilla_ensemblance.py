from Models.EAE_models.ensemblance_interface import EAE_interface
import tensorflow as tf


class Vanilla_ensemblance(EAE_interface):

    def __init__(self, dim, p_m, **kwargs):
        super(Vanilla_ensemblance, self).__init__(dim, **kwargs)
        self.fc1 = tf.keras.layers.Dense(self.h_dim)
        self.fc2 = tf.keras.layers.Dense(self.z_dim)
        self.fc3 = tf.keras.layers.Dense(self.h_dim)
        self.fc4 = tf.keras.layers.Dense(self.image_size)
        self._layers_ = [self.fc1, self.fc2, self.fc3,
                         self.fc4]
        self._masks_ = None
        self.p_m = p_m

    def initialize_masks(self):
        """
        Initialize masks for the model
        :return:
        """
        self._masks_ = [Vanilla_ensemblance.get_mask(i.get_weights()[0].shape,
                                                     p=self.p_m)
                        for i in self._layers_]

    def apply_masks(self):
        """
        Apply masks to all layers of the model
        :return:
        """
        for l, m in zip(self._layers_, self._masks_):
            new_weights = Vanilla_ensemblance.apply_mask(m,
                                                         l.get_weights())
            l.set_weights(new_weights)

    def encode(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        h = tf.nn.sigmoid(x)
        h = tf.nn.relu(self.fc1(x))
        return self.fc2(h)

    def decode(self, z):
        """
        Decode without activation function
        :param z: latent space
        :return: output model
        """
        h = tf.nn.relu(self.fc3(z))
        return tf.nn.sigmoid(self.fc4(h))

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