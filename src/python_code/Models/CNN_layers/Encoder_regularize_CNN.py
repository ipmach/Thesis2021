import tensorflow as tf


class EncoderCNN_Regularize(tf.keras.layers.Layer):

    def __init__(self, filters):
        """
        CNN encoder layers (tensorflow 2 book)
        :param filters: list filters
        """
        super(EncoderCNN_Regularize, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3,
                                            strides=1, activation='relu',
                                            padding='same',
                                            activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.conv2 = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3,
                                            strides=1, activation='relu',
                                            padding='same',
                                            activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.conv3 = tf.keras.layers.Conv2D(filters=filters[2], kernel_size=3,
                                            strides=1, activation='relu',
                                            padding='same',
                                            activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.reshape1 = tf.keras.layers.Reshape((1, 4 * 4 * 32))

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
        # print(x.shape)
        x = self.reshape1(x)
        return x