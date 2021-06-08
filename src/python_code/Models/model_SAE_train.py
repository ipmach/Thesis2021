import tensorflow as tf
from Models.model_SAE import SAE
from Models.AE_train_interface import TrainInterface
import numpy as np
from Models.loss import Loss


class SAE_train(SAE, TrainInterface):

    def __init__(self, dim, num_epochs, batch_size,
                 learning_rate, db,  **kwargs):
        """
         Wrapper for the train of  Sparse Autoencoder
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        :param num_epochs: number of epochs in training
        :param batch_size: Bacth size in traiing
        :param learning_rate: Learning rate
        :param db: Db where to save data training
        :param kwargs: optional parameters for VAE
        """
        super(SAE_train, self).__init__(dim, **kwargs)
        self.init_train(num_epochs, batch_size, learning_rate, db)

        self.build(input_shape=(4, self.image_size))

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

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
                x_reconstruction = self.decode(self.encode(x))
                loss = Loss.MSE(x, x_reconstruction)
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