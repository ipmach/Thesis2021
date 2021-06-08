import tensorflow as tf
import json


class DecoderCNN(tf.keras.layers.Layer):

    def __init__(self, filters, p_m, get_mask, apply_mask,
                 path_json='src/python_code/settings.json'):
        """
        CNN decoder layers (tensorflow 2 book)
        :param filters: list filters
        :param path_json: path settings
        """
        settings = json.load(open(path_json))["Model"]
        hyperparameters = settings["decoder_cnn"][int(settings["size_use"])]
        super(DecoderCNN, self).__init__()
        self.get_mask = get_mask
        self.apply_mask = apply_mask
        self.conv1 = tf.keras.layers.Conv2D(filters=filters[2], kernel_size=3,
                                            strides=1, activation='relu',
                                            padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3,
                                            strides=1, activation='relu',
                                            padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3,
                                            strides=1, activation='relu',
                                            padding=hyperparameters["padding_last"])
        self.conv4 = tf.keras.layers.Conv2D(filters=int(hyperparameters["channels_last"]),
                                            kernel_size=3, strides=1, activation='sigmoid',
                                            padding='same')
        self.upsample = tf.keras.layers.UpSampling2D((2, 2))
        self.reshape2 = tf.keras.layers.Reshape((4, 4, 32))
        self._layers_ = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.p_m = p_m

    def initialize_masks(self):
        """
        Initialize masks for the model
        :return:
        """
        self._masks_ = []
        for i in self._layers_:
            self.get_mask(i.get_weights()[0].shape,
                                         p=self.p_m)

    def apply_masks(self):
        """
        Apply masks to all layers of the model
        :return:
        """
        for l, m in zip(self._layers_, self._masks_):
            new_weights = self.apply_mask(m, l.get_weights())
            l.set_weights(new_weights)

    def call(self, encoded):
        x = self.reshape2(encoded)
        x = self.conv1(x)
        x = self.upsample(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.upsample(x)
        # print(x.shape)
        x = self.conv3(x)
        x = self.upsample(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        return x