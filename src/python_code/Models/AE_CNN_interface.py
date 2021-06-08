from Models.CNN_layers.Encoder_CNN import EncoderCNN
from Models.CNN_layers.Decoder_CNN import DecoderCNN
from Models.AAE_interface import AE


class AE_CNN(AE):

    def __init__(self, filters, z_dim,
                 encode_=EncoderCNN,
                 decoder_=DecoderCNN):
        """
        CNN interface
        :param filters: filters list
        :param z_dim: latent dimension
        :param encode_: layer cnn encoder
        :param decode_: layer cnn decoder
        """
        super(AE_CNN, self).__init__([1, z_dim, 1])
        self.filters = filters
        self.preformat = lambda x: x
        self.postformat = lambda x: x
        self.encoder_ = encode_(filters)
        self.decoder_ = decoder_(filters)

