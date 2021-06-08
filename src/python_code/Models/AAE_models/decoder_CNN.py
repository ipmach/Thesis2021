import tensorflow as tf
import json


class DecoderCNN(tf.keras.Model):

    def __init__(self, filters,
                 path_json='src/python_code/settings.json'):
        super(DecoderCNN, self).__init__()
        """
        CNN decoder layers (tensorflow 2 book)
        :param filters: list filters
        :param path_json: path settings
        """
        settings = json.load(open(path_json))["Model"]
        hyperparameters = settings["decoder_cnn"][int(settings["size_use"])]
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
        self.fc1 = tf.keras.layers.Dense(4*4*32)


    def call(self, encoded):
        x = self.fc1(encoded)
        x = self.reshape2(x)
        x = self.conv1(x)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.upsample(x)
        x = self.conv4(x)
        return x