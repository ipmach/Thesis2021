from Models.AAE_models import encoder, decoder, discriminator
from Models.AAE_interface import AE


class AAE(AE):

    def __init__(self, dim, num_labels, dropout=0.5, regularize=True, **kwargs):
        """
        Wrapper for the Adversal AutoEncoder (AAE)
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        :param num_labels: Number of labels for regularization
        :param dropout: Noise dropout [0,1]
        """
        # Define the hyperparameters
        super(AAE, self).__init__(dim, **kwargs)
        self.num_labels = num_labels
        self.dropout = dropout
        # Initialize the models
        self.encode_ = encoder.Encoder([self.h_dim, self.z_dim],
                                       self.dropout)
        self.decode_ = decoder.Decoder([self.h_dim, self.image_size],
                                       self.dropout)
        self.discriminator_ = discriminator.Discriminator([self.h_dim],
                                                         self.dropout)
        # Build the models
        self.encode_.build(input_shape=(4, self.image_size))
        self.decode_.build(input_shape=(4, self.z_dim))
        if regularize:
            self.discriminator_.build(input_shape=(4, self.z_dim + \
                                                  self.num_labels))
        else:
            self.discriminator_.build(input_shape=(4, self.z_dim))

    def load_weights_model(self, list_path):
        """
        Load the weights of the model
        :param path_encoder: path of the encoder weights (.h5)
        :param path_decoder: path of the decoder weights (.h5)
        :param path_discriminator: path of the discriminator weights (.h5)
        :return:
        """
        [path_encoder, path_decoder, path_discriminator] = list_path
        self.encode_.load_weights(path_encoder)
        self.decode_.load_weights(path_decoder)
        self.discriminator_.load_weights(path_discriminator)

    def save_weights_model(self, list_path):
        """
        Save the weights of the model
        """
        [path_encoder, path_decoder, path_discriminator] = list_path
        self.encode_.save_weights(path_encoder)
        self.decode_.save_weights(path_decoder)
        self.discriminator_.save_weights(path_discriminator)

    def encode(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        return self.encode_(x)

    def discriminator(self, x):
        """
        Discriminator input
        :param x: input
        :return: input in the latent space
        """
        return self.discriminator_(x)

    def decode(self, z):
        """
        Decode with activation function sigmoid
        :param z: latent space
        :return: output model
        """
        return self.decode_(z)

    def call_(self, inputs, training=None, mask=None, index=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :return
        """
        return self.decode_(self.encode_(inputs))