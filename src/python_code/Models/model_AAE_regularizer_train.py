import tensorflow as tf
from Models.model_AAE import AAE
from tqdm import tqdm
import numpy as np
from Models.AAE_models.loss import loss_AAE
import random


class AAE_train_re(AAE):
    # Inspire by https://github.com/Mind-the-Pineapple/adversarial-autoencoder

    def __init__(self, dim, num_epochs, batch_size,
                 learning_rate, db,  num_labels,
                 train_buf=60000, dropout=0.5, **kwargs):
        """
        Wrapper for the Adversal AutoEncoder (AAE)
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        :param num_labels: Number of labels for regularization
        :param dropout: Noise dropout [0,1]
        """
        super(AAE_train_re, self).__init__(dim, num_labels,
                                        dropout=dropout, **kwargs)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.db = db
        self.train_buf = train_buf
        self.build(input_shape=(4, self.image_size))

    @staticmethod
    def create_fake(value1, value2, size, z_dim,
                    mu_1, mu_2, std_1, std_2):
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
        part_1 = int(size / 2)
        part_2 = size - part_1
        a = tf.random.normal([part_1, z_dim], mean=mu_1, stddev=std_1)
        b = tf.random.normal([part_2, z_dim], mean=mu_2, stddev=std_2)
        labels = tf.concat([np.ones(part_1) * value1,
                            np.ones(part_2) * value2], -1)
        data = tf.concat([a, b], 0)
        d = list(zip(data, labels))
        random.shuffle(d)
        data, labels = list(zip(*d))
        data = tf.convert_to_tensor(data)
        labels = tf.convert_to_tensor(labels)
        return data, labels

    def train_step(self, batch_x, batch_y):
        # Autoencoder
        with tf.GradientTape() as ae_tape:
            encoder_output = self.encode_(batch_x, training=True)
            decoder_output = self.decode_(encoder_output, training=True)
            # Autoencoder loss
            ae_loss = self.loss.autoencoder_loss(batch_x, decoder_output)
        ae_grads = ae_tape.gradient(ae_loss,
                                    self.encode_.trainable_variables + \
                                    self.decode_.trainable_variables)
        self.ae_optimizer.apply_gradients(zip(ae_grads,
                                         self.encode_.trainable_variables + \
                                         self.decode_.trainable_variables))
        # Discriminator
        with tf.GradientTape() as dc_tape:
            ################
            data, labels = AAE_train_re.create_fake(self.value_1, self.value_2, batch_x.shape[0],
                                       self.z_dim, self.mu_1, self.mu_2, self.std_1, self.std_2)
            labels = tf.one_hot(tf.dtypes.cast(labels, tf.int32), 10)
            real_distribution = tf.concat([data, labels], -1)
            #################

            encoder_output = self.encode_(batch_x, training=True)

            dc_real = self.discriminator_(real_distribution, training=True)

            ################
            batch_y = tf.one_hot(tf.dtypes.cast(batch_y, tf.int32), 10)
            encoder_output = tf.concat([encoder_output, batch_y], -1)
            ################
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
            encoder_output = self.encode_(batch_x, training=True)
            encoder_output = tf.concat([encoder_output, batch_y], -1)
            dc_fake = self.discriminator_(encoder_output, training=True)

            # Generator loss
            gen_loss = self.loss.generator_loss(dc_fake)

        gen_grads = gen_tape.gradient(gen_loss, self.encode_.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.encode_.trainable_variables))

        return ae_loss, dc_loss, dc_acc, gen_loss

    def train(self, x_train, y_train, model_save,
              model_name,  value_1, value_2):
        """
        Start training
        :param x_train: data training
        :param model_save: path to save model
        :param model_name: name model
        :param value_1: label 1
        :param value_2: label 2
        :return:
        """
        self.value_1 = value_1
        self.value_2 = value_2
        reference_1 = [-3, 0, -3, 0]
        reference_2 = [3, 0, 3, 0]
        self.std_1 = np.ones(self.z_dim)
        self.std_2 = np.ones(self.z_dim)
        mu_1 = []
        mu_2 = []
        j = 0
        for _ in range(self.z_dim):
            mu_1.append(reference_1[j])
            mu_2.append(reference_2[j])
            j += 1
            if j >= len(reference_1):
                j = 0
        self.mu_1 = np.array(mu_1)
        self.mu_2 = np.array(mu_2)


        x_train = self.preprocessing(x_train)
        x_train = self.preformat(x_train)
        dataset = tf.data.Dataset.from_tensor_slices((x_train,
                                                      y_train))
        dataset = dataset.shuffle(buffer_size=self.train_buf)
        dataset = dataset.batch(self.batch_size)
        self.db.create_graph()
        self.db.upgrade_graph("Adversal Autoencoder", 0, self, x_train[:1])

        self.ae_optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.dc_optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.gen_optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.loss = loss_AAE()
        for epoch in tqdm(range(self.num_epochs), position=0):
            epoch_ae_loss_avg = tf.metrics.Mean()
            epoch_dc_loss_avg = tf.metrics.Mean()
            epoch_dc_acc_avg = tf.metrics.Mean()
            epoch_gen_loss_avg = tf.metrics.Mean()
            for batch, (batch_x, batch_y) in enumerate(dataset):
                ae_loss, dc_loss, dc_acc, gen_loss = self.train_step(batch_x, batch_y)
                epoch_ae_loss_avg(ae_loss)
                epoch_dc_loss_avg(dc_loss)
                epoch_dc_acc_avg(dc_acc)
                epoch_gen_loss_avg(gen_loss)
                self.db.insert_scalar_train(['General loss',
                                             'Accuracy Discriminator',
                                             'Loss Discriminato',
                                             'Loss Adversal AutoEncoder'],
                                            [epoch_gen_loss_avg.result(),
                                             epoch_dc_acc_avg.result(),
                                             epoch_dc_loss_avg.result(),
                                             epoch_ae_loss_avg.result()],
                                            epoch=epoch)
            # x_ = self(batch_x)
            # x_ = self.postprocessing(x_)
            # x_ = self.postformat(np.array(x_))[:5]
            # x = self.postprocessing(batch_x)
            # x = self.postformat(np.array(x))[:5]
            # self.db.insert_images_train(["Reconstructions " + model_name + " " + str(epoch)],
            #                             [x_], epoch)
            # self.db.insert_images_train(["Original " + model_name + " " + str(epoch)],
            #                             [x], epoch)
        self.encode_.save_weights(
            model_save + model_name + "_encoder" + '$z_' + str(self.z_dim) + "$h_" + str(self.h_dim) + ".h5")
        self.decode_.save_weights(
            model_save + model_name + "_decoder" + '$z_' + str(self.z_dim) + "$h_" + str(self.h_dim) + ".h5")
        self.discriminator_.save_weights(
            model_save + model_name + "_discriminator" + '$z_' + str(self.z_dim) + "$h_" + str(self.h_dim) + ".h5")
