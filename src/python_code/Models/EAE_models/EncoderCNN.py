import tensorflow as tf


class EncoderCNN(tf.keras.layers.Layer):

    def __init__(self, filters, p_m, get_mask, apply_mask):
        """
        CNN encoder layers (tensorflow 2 book)
        :param filters: list filters
        """
        super(EncoderCNN, self).__init__()
        self.get_mask = get_mask
        self.apply_mask = apply_mask
        self.conv1 = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3,
                                            strides=1, activation='relu',
                                            padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3,
                                            strides=1, activation='relu',
                                            padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=filters[2], kernel_size=3,
                                            strides=1, activation='relu',
                                            padding='same')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.reshape1 = tf.keras.layers.Reshape((1, 4 * 4 * 32))
        self._layers_ = [self.conv1, self.conv2, self.conv3]
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

    def call(self, input_features):
        # print(input_features.shape)
        x = self.conv1(input_features)
        x = self.pool(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.pool(x)
        # print(x.shape)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.reshape1(x)
        return x