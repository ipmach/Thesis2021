from Models.AE_train_interface import TrainInterface
from Models.model_VAE import VAE
from Models.loss import Loss
import tensorflow as tf


class VAE_train(VAE, TrainInterface):

    def __init__(self, dim, num_epochs, batch_size,
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
        super(VAE_train, self).__init__(dim, **kwargs)
        self.init_train(num_epochs, batch_size, learning_rate, db)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.likelihood = True

        self.build(input_shape=(4, self.image_size))

    def do_epoch(self, dataset, epoch):
        """
        Do operation epoch
        :param dataset:
        :param epoch:
        :return:
        """
        for step, x in enumerate(dataset):
            with tf.GradientTape() as tape:
                # Forward pass
                x_reconstruction_logits, mu, log_var = self(x, training=True)
                # Compute reconstruction loss and kl divergence
                reconstruction_loss = Loss.reconstruction_sigmoid(x,
                                                                  x_reconstruction_logits,
                                                                  self.batch_size)
                kl_div = Loss.KL_divergence(log_var, mu)
                # Backprop and optimize
                loss = reconstruction_loss + kl_div
                self.db.insert_scalar_train(['Total loss', 'KL divergence',
                                             'Reconstruction loss'],
                                            [loss, kl_div, reconstruction_loss],
                                            epoch=epoch)
            gradients = tape.gradient(loss, self.trainable_variables)
            for g in gradients:
                tf.clip_by_norm(g, 15)
            self.optimizer.apply_gradients(zip(gradients,
                                           self.trainable_variables))

    def do_save(self, model_save, model_name):
        """
        Save model
        :return:
        """
        self.save_weights_model([model_save + model_name + ".h5"])