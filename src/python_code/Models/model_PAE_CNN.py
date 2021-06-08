from Models.PAE_models.Encoder_CNN import EncoderCNN
from Models.PAE_models.Decoder_CNN import DecoderCNN
from Models.PAE_models.Bijecter import RealNVP
from sklearn.preprocessing import MinMaxScaler
from Models.AE_CNN_interface import AE_CNN
import joblib


class PAECNN(AE_CNN):

    def __init__(self, filters, dim, switch=False, **kwargs):
        """
        Wrapper for the Probabilistic AutoEncoder (PAE)
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        """
        # Define the hyperparameters
        super(PAECNN, self).__init__(filters, dim, **kwargs)
        # Initialize the models
        self.encoder_ = EncoderCNN(filters, self.z_dim)
        self.decoder_ = DecoderCNN(filters)
        self.b = RealNVP(num_coupling_layers=6, z_dim=self.z_dim)
        self.scaler = MinMaxScaler()
        # Use bijecter in incode or not
        self.switch = switch

        self.encoder_.build(input_shape=(None, self.img_shape[0],
                            self.img_shape[1], self.img_shape[2]))
        self.decoder_.build(input_shape=(None, self.z_dim))
        self.b.build(input_shape=(None, self.z_dim))
        self.encode_ = self.encoder_

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
        self.encoder_.load_weights(path_encoder)
        self.decoder_.load_weights(path_decoder)
        self.b.load_weights(path_discriminator)
        self.scaler = joblib.load(path_scaler)

    def save_weights_model(self, list_path):
        """
        Save the weights of the model
        """
        [path_encoder, path_decoder, path_discriminator, path_scaler] = list_path
        self.encoder_.save_weights(path_encoder)
        self.decoder_.save_weights(path_decoder)
        self.b.save_weights(path_discriminator)
        joblib.dump(self.scaler, path_scaler + 'scaler.pkl')

    def encode(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        if self.switch:
            z = self.encoder_(x)
            return self.bijecter(z)
        else:
            return self.encoder_(x)

    def decode(self, z):
        """
        Decode with activation function sigmoid
        :param z: latent space
        :return: output model
        """
        if self.switch:
           x = self.bijecter(z, inverse=True)
           return self.decoder_(x)
        else:
            return self.decoder_(z)

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
        return self.decoder_(self.encoder_(inputs))