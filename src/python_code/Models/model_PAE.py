from Models.PAE_models.Bijecter import RealNVP
from sklearn.preprocessing import MinMaxScaler
from Models.PAE_models.Encoder import Encoder
from Models.PAE_models.Decoder import Decoder
from Models.AAE_interface import AE
import joblib


class PAE(AE):

    def __init__(self, dim, switch=False, **kwargs):
        """
        Wrapper for the Probabilistic AutoEncoder (PAE)
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        """
        # Define the hyperparameters
        super(PAE, self).__init__(dim, **kwargs)
        # Initialize the models
        self.encode_ = Encoder([self.h_dim, self.z_dim])
        self.decode_ = Decoder([self.h_dim, self.image_size])
        self.b = RealNVP(num_coupling_layers=6, z_dim=self.z_dim)
        self.scaler = MinMaxScaler()
        # Build the models
        self.encode_.build(input_shape=(None, self.image_size))
        self.decode_.build(input_shape=(None, self.z_dim))
        self.b.build(input_shape=(None, self.z_dim))
        # Use bijecter in incode or not
        self.switch = switch


    def load_weights_model(self, list_path):
        """
        Load the weights of the model
        :param path_encoder: path of the encoder weights (.h5)
        :param path_decoder: path of the decoder weights (.h5)
        :param path_discriminator: path of the discriminator weights (.h5)
        :param path_scaler: path scaler (.pkl)
        :return:
        """
        [path_encoder, path_decoder, path_discriminator, path_scaler] = list_path
        self.encode_.load_weights(path_encoder)
        self.decode_.load_weights(path_decoder)
        self.b.load_weights(path_discriminator)
        self.scaler = joblib.load(path_scaler)

    def save_weights_model(self, list_path):
        """
        Save the weights of the model
        """
        [path_encoder, path_decoder, path_discriminator, path_scaler] = list_path
        self.encode_.save_weights(path_encoder)
        self.decode_.save_weights(path_decoder)
        self.b.save_weights(path_discriminator)
        joblib.dump(self.scaler, path_scaler + 'scaler.pkl')

    def encode(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        if self.switch:
            z = self.encode_(x)
            return self.bijecter(z)
        else:
            return self.encode_(x)

    def decode(self, z):
        """
        Decode with activation function sigmoid
        :param z: latent space
        :return: output model
        """
        if self.switch:
           x = self.bijecter(z, inverse=True)
           return self.decode_(x)
        else:
            return self.decode_(z)

    def bijecter(self, z, inverse=False):
        if inverse:
            b_data, _ = self.b.predict(z)
            return self.scaler.inverse_transform(b_data)
        else:
            b_data = self.scaler.transform(z)
            b_data, _ = self.b(b_data)
            return b_data

    def call_(self, inputs, training=None, mask=None, index=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :return
        """
        return self.decode_(self.encode_(inputs))