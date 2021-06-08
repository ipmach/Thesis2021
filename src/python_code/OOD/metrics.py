import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


class Metrics:

    @staticmethod
    def MSE(x, x_, db, dataset, epoch, num=None):
        aux = np.mean((x-x_)**2, axis=1)
        db.save_ood(dataset, aux, epoch, num=num)
        return np.mean(aux)

    @staticmethod
    def NormLikehood(x, db, dataset, epoch, dist):
        aux = tf.math.reduce_mean(dist.prob(x), axis=1)
        db.save_ood(dataset, aux, epoch)

    @staticmethod
    def Log_NormLikehood(x, db, dataset, epoch, dist):
        aux = tf.math.reduce_mean(dist.log_prob(x), axis=1)
        db.save_ood(dataset, aux, epoch)


class BinaryCross:

    def __init__(self):
        self.bc = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)

    def __call__(self, x, x_reconstruction,
                 db, dataset, epoch, num=None):
        aux = self.bc(x, x_reconstruction)
        db.save_ood(dataset, aux, epoch, num=num)
        return np.mean(aux)


class Mahalanobis:

    def __init__(self, x):
        self.cov = tf.linalg.inv(tfp.stats.covariance(x))
        self.cov = tf.reshape(self.cov, (self.cov.shape[1], self.cov.shape[1]))
        self.mu = tf.math.reduce_mean(x, axis=0)

    def operation(self, x):
        x_mu = tf.reshape(x - self.mu, (x.shape[0], -1))
        return tf.linalg.matmul(tf.linalg.matmul(x_mu, self.cov),
                                tf.transpose(x_mu))

    def __call__(self, x,
                 db, dataset, epoch, num=None):
        aux = tf.reshape(tf.map_fn(self.operation, x), -1)
        db.save_ood(dataset, aux, epoch, num=num)