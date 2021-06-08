import tensorflow as tf
from Models.model_DAE import DAE
from Models.AE_train_interface import TrainInterface
import numpy as np
from Models.loss import Loss


class DAE_train(DAE, TrainInterface):

    def __init__(self, dim, num_epochs, batch_size,
                 learning_rate, db,  **kwargs):
        """
         Wrapper for the train of  Denoising Autoencoder
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        :param num_epochs: number of epochs in training
        :param batch_size: Bacth size in traiing
        :param learning_rate: Learning rate
        :param db: Db where to save data training
        :param kwargs: optional parameters for VAE
        """
        super(DAE_train, self).__init__(dim, **kwargs)
        self.init_train(num_epochs, batch_size, learning_rate, db)

        self.build(input_shape=(4, self.image_size))

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def init_dataset(self, x_train, y_train):
        """
        Initialize dataset
        :param x_train:
        :param y_train:
        :return:
        """
        noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
        x_train_noise = x_train + noise
        dataset = tf.data.Dataset.from_tensor_slices((x_train_noise, x_train))
        return dataset.shuffle(self.batch_size * 5).batch(self.batch_size)

    def do_epoch(self, dataset, epoch):
        """
        Do operation epoch (Inside loop of epochs)
        :param dataset:
        :param epoch:
        :return:
        """
        loss_steps = []
        for step, (x_noise, x) in enumerate(dataset):
            with tf.GradientTape() as tape:
                x_reconstruction = self.decode(self.encode(x_noise))
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

