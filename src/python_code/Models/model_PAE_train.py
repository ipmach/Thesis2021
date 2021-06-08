from Models.AE_train_interface import TrainInterface
from tqdm.keras import TqdmCallback
from Models.model_PAE import PAE
from Models.loss import Loss
import tensorflow as tf
import numpy as np
import joblib


class PAE_train(PAE, TrainInterface):

    def __init__(self, dim, num_epochs, batch_size,
                 learning_rate, db, b_epochs=5, b_batch=256,
                 b_learning=0.0001, b_split=0.2, **kwargs):
        """
         Wrapper for the train of  Probabilistic Autoencoder (PAE)
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        :param num_epochs: number of epochs in training
        :param batch_size: Bacth size in traiing
        :param learning_rate: Learning rate
        :param db: Db where to save data training
        :param b_epochs: Bijecter epochs train
        :param b_batch: Bijecter batch size
        :param b_learning: Bijecter learning rate
        :param b_split: Bijecter split train-test [0,1]
        :param kwargs: optional parameters for VAE
        """
        super(PAE_train, self).__init__(dim, **kwargs)
        self.init_train(num_epochs, batch_size, learning_rate, db)

        self.build(input_shape=(4, self.image_size))
        self.b_epochs = b_epochs
        self.b_batch = b_batch
        self.b_learning = b_learning
        self.b_split = b_split

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
                encoder_output = self.encode_(x, training=True)
                decoder_output = self.decode_(encoder_output, training=True)
                loss = Loss.MSE(x, decoder_output)
                loss_steps.append(loss)
            tape = tape.gradient(loss,
                                 self.encode_.trainable_variables + \
                                 self.decode_.trainable_variables)
            self.optimizer.apply_gradients(zip(tape,
                                               self.encode_.trainable_variables + \
                                               self.decode_.trainable_variables))
        self.db.insert_scalar_train(['Reconstruction loss'],
                                    [np.mean(loss_steps)],
                                    epoch=epoch)

    def do_after_train(self, x_train, y_train):
        """
        Do after last epoch
        :param y_train:
        :return:
        """
        encoder_output = self.encode_(x_train)
        encoder_output = self.scaler.fit_transform(encoder_output)
        self.b.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.b_learning))

        history = self.b.fit(
            encoder_output, batch_size=self.b_batch, epochs=self.b_epochs,
            verbose=2, validation_split=self.b_split,
            callbacks=[TqdmCallback(verbose=2)]
        )
        # Temporal ## NEED TO BE SOLVE
        for i in range(len(history.history)):
            self.db.insert_scalar_train(['Bijector loss train'],
                                        [history.history["loss"][i]],
                                        epoch=i)
            self.db.insert_scalar_train(['Bijector loss test'],
                                        [history.history["val_loss"][i]],
                                        epoch=i)

    def do_save(self, model_save, model_name):
        """
        Save model
        :return:
        """
        self.encode_.save_weights(
            model_save + model_name + "_encoder" + '$z_' + str(self.z_dim) + "$h_" + str(self.h_dim) + ".h5")
        self.decode_.save_weights(
            model_save + model_name + "_decoder" + '$z_' + str(self.z_dim) + "$h_" + str(self.h_dim) + ".h5")
        self.b.save_weights(
            model_save + model_name + "_b" + '$z_' + str(self.z_dim) + "$h_" + str(self.h_dim) + ".h5")
        joblib.dump(self.scaler, model_save + model_name + 'scaler.pkl')


