from Models.AE_train_interface import TrainInterface
from Models.model_SAE_CNN import SAECNN
from Models.loss import BinaryCross
import tensorflow as tf
import numpy as np


class SAECNN_train(SAECNN, TrainInterface):

    def __init__(self, filters, dim, num_epochs, batch_size,
                 learning_rate, db,  **kwargs):
        """
         Wrapper for the train of  Sparse Autoencoder CNN(Vanilla)
        :param dim: latent space dimension
        :param num_epochs: number of epochs in training
        :param batch_size: Bacth size in training
        :param learning_rate: Learning rate
        :param db: Db where to save data training
        :param kwargs: optional parameters for VAE
        """
        super(SAECNN_train, self).__init__(filters, dim, **kwargs)
        self.init_train(num_epochs, batch_size, learning_rate, db)

        self.build(input_shape=(None, self.img_shape[0],
                                self.img_shape[1], self.img_shape[2]))

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.bc = BinaryCross()

    def do_epoch(self, dataset, epoch):
        """
        Do operation epoch
        :param dataset:
        :param epoch:
        :return:
        """
        loss_steps = []
        for step, x in enumerate(dataset):
            with tf.GradientTape() as tape:
                x_reconstruction = self(x)
                loss = self.bc(x, x_reconstruction)
                loss_steps.append(loss)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients,
                                          self.trainable_variables))
        self.db.insert_scalar_train(['Reconstruction loss'],
                                    [np.mean(loss_steps)],
                                    epoch=epoch)

    def do_save(self, model_save, model_name):
        """
        Save model
        :return:
        """
        self.save_weights_model([model_save + model_name + ".h5"])
