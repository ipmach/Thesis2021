from Models.AAE_models import encoder_CNN, decoder_CNN, discriminator_CNN
from Models.AE_CNN_interface import AE_CNN


class AAECNN(AE_CNN):

    def __init__(self, filters, dim, num_labels=10, regularize=False, **kwargs):
        """
        Wrapper for the Adversal AutoEncoder (AAE)
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        :param num_labels: Number of labels for regularization
        :param dropout: Noise dropout [0,1]
        """
        # Define the hyperparameters
        [z_dim, h_dim] = dim
        super(AAECNN, self).__init__(filters, z_dim, **kwargs)
        self.num_labels = num_labels
        # Initialize the models
        self.encoder_ = encoder_CNN.EncoderCNN(filters, z_dim)
        self.encode_ = self.encoder_
        self.decoder_ = decoder_CNN.DecoderCNN(filters)
        self.discriminator_ = discriminator_CNN.Discriminator([h_dim])
        # Build the models
        self.encoder_.build(input_shape=(None, self.img_shape[0],
                            self.img_shape[1], self.img_shape[2]))
        self.decoder_.build(input_shape=(None, z_dim))
        if regularize:
            self.discriminator_.build(input_shape=(4, z_dim + \
                                                  self.num_labels))
        else:
            self.discriminator_.build(input_shape=(4, z_dim))

    def encode(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        return self.encoder_(x)

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
        return self.decoder_(z)

    def call_(self, inputs, training=None, mask=None, index=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :return
        """
        return self.decode(self.encode(inputs))

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