import tensorflow as tf
from Models.AAE_interface import AE


class DAE(AE):

    def __init__(self, dim, **kwargs):
        super(DAE, self).__init__(dim, **kwargs)
        #if self.h_dim < self.image_size:
        #    raise Exception("h_dim must be bigger than image_size")
        self.fc1 = tf.keras.layers.Dense(self.h_dim)
        self.fc2 = tf.keras.layers.Dense(self.z_dim)
        self.fc3 = tf.keras.layers.Dense(self.h_dim)
        self.fc4 = tf.keras.layers.Dense(self.image_size)

    def encode(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
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

    def call_(self, inputs, training=None, mask=None, index=None):
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