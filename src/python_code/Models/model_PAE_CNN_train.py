from Models.AE_train_interface import TrainInterface
from Models.model_PAE_CNN import PAECNN
from tqdm.keras import TqdmCallback
from Models.loss import BinaryCross
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import joblib
import json


class PAECNN_train(PAECNN, TrainInterface):

    def __init__(self, filters, dim, num_epochs, batch_size,
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
        super(PAECNN_train, self).__init__(filters, dim, **kwargs)
        self.init_train(num_epochs, batch_size, learning_rate, db)

        self.b_epochs = b_epochs
        self.b_batch = b_batch
        self.b_learning = b_learning
        self.b_split = b_split

        self.likelihood = True
        self.loglikelihood = True
        self.mahalanobis = True

        self.build(input_shape=(None, self.img_shape[0],
                                self.img_shape[1], self.img_shape[2]))

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.bc = BinaryCross()

    def train(self, x_train, model_save, model_name,
              ood_data=None, ood_op=None, x_test=None, y_train=None, ood_i=1.1):
        """
        Start training
        :param x_train: data training
        :param model_save: path to save model
        :param model_name: name model
        :return:
        """
        settings = json.load(open('src/python_code/settings.json'))["Train_script"]
        save_image = settings["save_images"]
        self.model_name = model_name
        self.do_before_train(x_train, y_train)
        x_train = self.pre_model(x_train)
        if ood_data is not None and x_test is not None:
            ood_op.initialize_data(self, ood_data, x_test=x_test)
        dataset = self.init_dataset(x_train, y_train)
        self.db.create_graph()
        self.db.upgrade_graph(self.model_name, 0, self, x_train[:1])
        lim = ood_i
        z = 0
        j = 0
        for epoch in tqdm(range(self.num_epochs)):
            self.do_epoch(dataset, epoch)
            j += 1
            if ood_data is not None and x_test is not None and j >= lim:
                self.do_after_train(x_train, y_train)
                print("HIII")
                self.switch = True  # Allow access to second LS
                ood_op.operate_data(self, self.db, epoch,
                                    x_train=x_train, disc=self.disc,
                                    likelihood=self.likelihood, svm1=self.svm1,
                                    loglikelihood=self.loglikelihood, mahalanobis=self.mahalanobis)
                self.switch = False  # Disable access to second LS
                j = 0
                z += 1
                lim = np.floor(1.1**z)
            if epoch % save_image == 0:
                self.do_img(x_train, epoch)
        self.do_after_train(x_train, y_train)
        self.do_save(model_save, model_name)


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
            tape = tape.gradient(loss,
                                 self.encoder_.trainable_variables + \
                                 self.decoder_.trainable_variables)
            self.optimizer.apply_gradients(zip(tape,
                                               self.encoder_.trainable_variables + \
                                               self.decoder_.trainable_variables))
        self.db.insert_scalar_train(['Reconstruction loss'],
                                    [np.mean(loss_steps)],
                                    epoch=epoch)

    def do_after_train(self, x_train, y_train):
        """
        Do after last epoch
        :param y_train:
        :return:
        """
        encoder_output = self.encode_.predict(x_train, batch_size=100)
        encoder_output = tf.reshape(encoder_output, (-1, self.z_dim))
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
        self.encoder_.save_weights(
            model_save + model_name + "_encoder" + '$z_' + str(self.z_dim) + "$h_" + str(self.h_dim) + ".h5")
        self.decoder_.save_weights(
            model_save + model_name + "_decoder" + '$z_' + str(self.z_dim) + "$h_" + str(self.h_dim) + ".h5")
        self.b.save_weights(
            model_save + model_name + "_b" + '$z_' + str(self.z_dim) + "$h_" + str(self.h_dim) + ".h5")
        joblib.dump(self.scaler, model_save + model_name + 'scaler.pkl')