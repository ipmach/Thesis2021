import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

# https://keras.io/examples/generative/real_nvp/

output_dim = 256
reg = 0.01


def Coupling(input_shape):
    input = tf.keras.layers.Input(shape=input_shape)

    t_layer_1 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(input)
    t_layer_2 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(t_layer_1)
    t_layer_3 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(t_layer_2)
    t_layer_4 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(t_layer_3)
    t_layer_5 = tf.keras.layers.Dense(
        input_shape, activation="linear", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(t_layer_4)

    s_layer_1 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(input)
    s_layer_2 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(s_layer_1)
    s_layer_3 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(s_layer_2)
    s_layer_4 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(s_layer_3)
    s_layer_5 = tf.keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(s_layer_4)

    return tf.keras.Model(inputs=input, outputs=[s_layer_5, t_layer_5])


class RealNVP(tf.keras.Model):
    # https://keras.io/examples/generative/real_nvp/

    def __init__(self, num_coupling_layers, z_dim):
        super(RealNVP, self).__init__()

        self.num_coupling_layers = num_coupling_layers
        filter = list(np.rot90(np.eye(z_dim)))
        filter = [list(i) for i in filter]
        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0. for _ in range(z_dim)],
            scale_diag=[1. for _ in range(z_dim)]
        )
        self.masks = np.array(
            filter * (num_coupling_layers // 2), dtype="float32"
        )
        self.mask = tf.convert_to_tensor(self.masks)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.layers_list = [Coupling(z_dim) for i in range(num_coupling_layers)]

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self, x, training=True):
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s, [1])

        return x, log_det_inv

    # Log likelihood of the normal distribution plus the log determinant of the jacobian.

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        #log_likelihood = logdet
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.log_loss(data)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}
