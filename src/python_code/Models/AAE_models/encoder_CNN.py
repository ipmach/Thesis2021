import tensorflow as tf


class EncoderCNN(tf.keras.Model):

    def __init__(self, filters, z_dim):
        """
        CNN encoder layers (tensorflow 2 book)
        :param filters: list filters
        """
        super(EncoderCNN, self).__init__()
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
        self.reshape1 = tf.keras.layers.Reshape((1, 4*4*32))
        self.fc1 = tf.keras.layers.Dense(z_dim)


    def call(self, input_features):
        x = self.conv1(input_features)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.reshape1(x)
        x = self.fc1(x)
        return x
