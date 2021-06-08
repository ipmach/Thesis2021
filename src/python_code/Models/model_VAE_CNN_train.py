from Models.AE_train_interface import TrainInterface
from Models.loss import Loss, BinaryCross
from Models.model_VAE_CNN import VAECNN
import tensorflow as tf
import numpy as np


class VAECNN_train(VAECNN, TrainInterface):

    def __init__(self, filters, dim, num_epochs, batch_size,
                 learning_rate, db,  **kwargs):
        """
         Wrapper for the train of  Varational Autoencoder (VAE)
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        :param num_epochs: number of epochs in training
        :param batch_size: Bacth size in traiing
        :param learning_rate: Learning rate
        :param db: Db where to save data training
        :param kwargs: optional parameters for VAE
        """
        super(VAECNN_train, self).__init__(filters, dim, **kwargs)
        self.init_train(num_epochs, batch_size, learning_rate, db)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.likelihood = True
        self.loglikelihood = True
        self.mahalanobis = True

        self.build(input_shape=(None, self.img_shape[0],
                                self.img_shape[1], self.img_shape[2]))
        self.bc = BinaryCross()

    def do_epoch(self, dataset, epoch):
        """
        Do operation epoch
        :param dataset:
        :param epoch:
        :return:
        """
        loss_steps = []
        recons_steps = []
        kl_steps = []
        bc_steps = []
        for step, x in enumerate(dataset):
            with tf.GradientTape() as tape:
                # Forward pass
                x_reconstruction_logits, mu, log_var = self(x, training=True)
                x = tf.cast(x, tf.float32)  # Necessary for sigmoid reconstruction
                # Binary crossentropy just for comparison
                bc_loss = self.bc(x, tf.math.sigmoid(x_reconstruction_logits))
                reconstruction_loss = Loss.reconstruction_sigmoid(x, x_reconstruction_logits,
                                                                  self.batch_size)
                kl_div = Loss.KL_divergence(mu, log_var)

                loss = reconstruction_loss + kl_div
                loss_steps.append(loss)
                bc_steps.append(bc_loss)
                recons_steps.append(reconstruction_loss)
                kl_steps.append(kl_div)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients,
                                               self.trainable_variables))

        #print(np.mean(kl_steps), np.mean(loss_steps))
        self.db.insert_scalar_train(['Total loss', 'KL divergence',
                                     'Reconstruction Sigmoid', 'Reconstruction loss'],
                                    [np.mean(loss_steps), np.mean(kl_steps),
                                     np.mean(recons_steps), np.mean(bc_steps)],
                                    epoch=epoch)

    def do_save(self, model_save, model_name):
        """
        Save model
        :return:
        """
        self.save_weights_model([model_save + model_name + ".h5"])