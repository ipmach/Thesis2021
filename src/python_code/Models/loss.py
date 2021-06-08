import tensorflow as tf


class Loss:

    @staticmethod
    def MSE(x, x_):
        return tf.math.reduce_mean(
                                tf.keras.losses.mean_squared_error(x, x_))

    @staticmethod
    def reconstruction_sigmoid(x, x_reconstruction_logits,
                               batch_size):
        """
        Reconstruction loss with sigmoid cross entropy
        :param x: original
        :param x_reconstruction_logits: reconstruction
        :param batch_size: size of the batch
        :return: loss
        """
        reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=x, logits=x_reconstruction_logits)
        reconstruction_loss = tf.reduce_sum(reconstruction_loss) / batch_size
        return reconstruction_loss

    @staticmethod
    def KL_divergence(log_var, mu):
        """
        KL divergence
        :param log_var: log of std
        :param mu: mean
        :return: loss
        """
        kl_div = -0.5 * tf.reduce_sum(1. + log_var - tf.square(mu) -
                                      tf.exp(log_var), axis=-1)
        return tf.reduce_mean(kl_div)


class BinaryCross:

    def __init__(self):
        self.bc = tf.keras.losses.BinaryCrossentropy()

    def __call__(self, x, x_reconstruction):
        return self.bc(x, x_reconstruction)

