from Models.AE_train_interface import TrainInterface
from Models.AAE_models.loss import loss_AAE
from Models.model_AAE_CNN import AAECNN
from Models.loss import BinaryCross
import tensorflow as tf
import numpy as np


class AAECNN_train(AAECNN, TrainInterface):
    # Inspire by https://github.com/Mind-the-Pineapple/adversarial-autoencoder

    def __init__(self, filters, dim, num_epochs, batch_size,
                 learning_rate, db,  num_labels,
                 train_buf=60000, dropout=0.5, **kwargs):
        """
        Wrapper for the Adversal AutoEncoder Regularizer(AAE)
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        :param num_labels: Number of labels for regularization
        :param dropout: Noise dropout [0,1]
        """
        super(AAECNN_train, self).__init__(filters, dim, num_labels,
                                           regularize=False, **kwargs)
        self.init_train(num_epochs, batch_size, learning_rate, db)
        self.train_buf = train_buf
        self.loss = loss_AAE()
        self.disc = True
        self.likelihood = True
        self.loglikelihood = True
        self.mahalanobis = True

        self.build(input_shape=(None, self.img_shape[0],
                                self.img_shape[1], self.img_shape[2]))

        self.ae_optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.dc_optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.gen_optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

        self.bc = BinaryCross()

    @staticmethod
    def create_fake(size, z_dim, mu, std):
        """
        Create two fake independent gaussian distributions
        :param value1: label first distribution
        :param value2: label second distribution
        :param size: Size of sample
        :param z_dim: Size of the latent space
        :param mu_1: mean first distribution
        :param mu_2: mean second distribution
        :param std_1: standard desviation first distribution
        :param std_2: standart desviation second distribution
        :return: data and labels
        """
        return tf.random.normal([size, z_dim], mean=0.0, stddev=1.0)

    def train_step(self, batch_x):
        # Autoencoder
        with tf.GradientTape() as ae_tape:
            encoder_output = self.encoder_(batch_x, training=True)
            decoder_output = self.decoder_(encoder_output, training=True)
            # Autoencoder loss
            ae_loss = self.bc(batch_x, decoder_output)
        ae_grads = ae_tape.gradient(ae_loss,
                                    self.encoder_.trainable_variables + \
                                    self.decoder_.trainable_variables)
        self.ae_optimizer.apply_gradients(zip(ae_grads,
                                              self.encoder_.trainable_variables + \
                                              self.decoder_.trainable_variables))
        # Discriminator
        with tf.GradientTape() as dc_tape:
            ################
            real_distribution = AAECNN_train.create_fake(batch_x.shape[0],
                                                         self.z_dim, self.mu, self.std)
            #################

            encoder_output = self.encoder_(batch_x, training=True)

            dc_real = self.discriminator_(real_distribution, training=True)

            dc_fake = self.discriminator_(encoder_output, training=True)

            # Discriminator Loss
            dc_loss = self.loss.discriminator_loss(dc_real, dc_fake)
            # Discriminator Acc
            dc_acc = self.loss.accuracy(tf.concat([tf.ones_like(dc_real),
                                                   tf.zeros_like(dc_fake)], axis=0),
                                        tf.concat([dc_real, dc_fake], axis=0))

        dc_grads = dc_tape.gradient(dc_loss, self.discriminator_.trainable_variables)
        self.dc_optimizer.apply_gradients(zip(dc_grads,
                                              self.discriminator_.trainable_variables))
        # Generator (Encoder)
        with tf.GradientTape() as gen_tape:
            encoder_output = self.encoder_(batch_x, training=True)
            dc_fake = self.discriminator_(encoder_output, training=True)

            # Generator loss
            gen_loss = self.loss.generator_loss(dc_fake)

        gen_grads = gen_tape.gradient(gen_loss, self.encoder_.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.encoder_.trainable_variables))

        return ae_loss, dc_loss, dc_acc, gen_loss

    def do_before_train(self, x_train, y_train):
        """
        Do before first epoch
        :param y_train:
        :return:
        """
        reference_1 = [0, 1]
        self.std = np.ones(self.z_dim)
        mu = []
        j = 0
        for _ in range(self.z_dim):
            mu.append(reference_1[j])
            j += 1
            if j >= len(reference_1):
                j = 0
        self.mu = np.array(mu)

    def init_dataset(self, x_train, y_train):
        """
        Initialize dataset
        :param x_train:
        :param y_train:
        :return:
        """
        dataset = tf.data.Dataset.from_tensor_slices(x_train)
        dataset = dataset.shuffle(buffer_size=self.train_buf)
        return  dataset.batch(self.batch_size)

    def do_epoch(self, dataset, epoch):
        """
        Do operation epoch
        :param dataset:
        :param epoch:
        :return:
        """
        epoch_ae_loss_avg = tf.metrics.Mean()
        epoch_dc_loss_avg = tf.metrics.Mean()
        epoch_dc_acc_avg = tf.metrics.Mean()
        epoch_gen_loss_avg = tf.metrics.Mean()
        for batch, batch_x in enumerate(dataset):
            ae_loss, dc_loss, dc_acc, gen_loss = self.train_step(batch_x)
            epoch_ae_loss_avg(ae_loss)
            epoch_dc_loss_avg(dc_loss)
            epoch_dc_acc_avg(dc_acc)
            epoch_gen_loss_avg(gen_loss)
        self.db.insert_scalar_train(['General loss',
                                     'Accuracy Discriminator',
                                     'Loss Discriminator',
                                     'Loss Adversal AutoEncoder'],
                                    [epoch_gen_loss_avg.result(),
                                     epoch_dc_acc_avg.result(),
                                     epoch_dc_loss_avg.result(),
                                     epoch_ae_loss_avg.result()],
                                    epoch=epoch)

    def do_save(self, model_save, model_name):
        """
        Save model
        :return:
        """
        paths = [model_save + model_name + "_encoder" + '$z_' + str(self.z_dim) + "$h_" + str(self.h_dim) + ".h5",
                 model_save + model_name + "_decoder" + '$z_' + str(self.z_dim) + "$h_" + str(self.h_dim) + ".h5",
                 model_save + model_name + "_discriminator" + '$z_' + str(self.z_dim) + "$h_" + str(self.h_dim) + ".h5"]
        self.save_weights_model(paths)