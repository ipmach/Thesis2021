from Models.AE_train_interface import TrainInterface
from Models.loss import Loss, BinaryCross
from Models.model_EAE import RandNet
import tensorflow as tf
from tqdm import tqdm
import numpy as np


class RandNet_train(RandNet, TrainInterface):

    def __init__(self, dim, p_m, num, num_epochs, batch_size,
                 learning_rate, db, cnn=False ,filters=None, **kwargs):
        """
         Wrapper for the train of  Vanilla Autoencoder (Vanilla)
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        :param p_m: Mask densitiy [0,1]
        :param num: number of models
        :param num_epochs: number of epochs in training
        :param batch_size: Bacth size in traiing
        :param learning_rate: Learning rate
        :param db: Db where to save data training
        :param kwargs: optional parameters for VAE
        """
        super(RandNet_train, self).__init__(dim, p_m, num, cnn=cnn,filters=filters, **kwargs)
        self.init_train(num_epochs, batch_size, learning_rate, db)

        if cnn:
            self.build(input_shape=(None, self.img_shape[0],
                                    self.img_shape[1], self.img_shape[2]))
        else:
            self.build(input_shape=(4, self.image_size))
        self.initialize_masks()

        self.optimizers = [tf.keras.optimizers.RMSprop(0.001)
                           for _ in range(self.num_models)]

        self.bc = BinaryCross()

    def do_epoch(self, dataset, epoch):
        """
        Do operation epoch
        :param dataset:
        :param epoch:
        :return:
        """
        losses = [[] for _ in range(self.num_models)]
        for step, x in enumerate(dataset):
            self.apply_masks()
            loss = []
            for i in range(self.num_models):
                with tf.GradientTape() as tape:
                    x_reconstruction = self.decode(self.encode(x, index=i), index=i)
                    l = self.bc(x, x_reconstruction)
                    loss.append(l)
                    losses[i].append(l)
                gradients = tape.gradient(l, self[i].trainable_variables)
                self.optimizers[i].apply_gradients(zip(gradients,
                                                       self[i].trainable_variables))
        self.db.insert_scalar_train(['Reconstruction loss'],
                                    [np.mean(losses)],
                                    epoch=epoch)

        for i in range(self.num_models):
            self.db.insert_scalar_train(['Reconstruction loss ' + str(i)],
                                        [np.mean(losses[i])],
                                        epoch=epoch)

    def do_save(self, model_save, model_name):
        """
        Save model
        :return:
        """
        for i in range(self.num_models):
            self[i].save_weights(model_save + model_name + "_" + str(i) + ".h5")

    def train(self, x_train, model_save, model_name, save_image=20,
              ood_data=None, ood_op=None, x_test=None, y_train=None, ood_i=1.1):
        """
        Start training
        :param x_train: data training
        :param model_save: path to save model
        :param model_name: name model
        :return:
        """
        self.do_before_train(x_train, y_train)
        x_train = self.pre_model(x_train)
        if ood_data is not None and x_test is not None:
            ood_op.initialize_data(self, ood_data, x_test=x_test)
        dataset = self.init_dataset(x_train, y_train)
        self.db.create_graph()
        self.db.upgrade_graph(model_name, 0, self, x_train[:1])
        lim = ood_i
        z = 0
        j = 0
        for epoch in tqdm(range(self.num_epochs)):
            self.do_epoch(dataset, epoch)
            j += 1
            if ood_data is not None and x_test is not None and j >= lim:
                ood_op.operate_data_ensembles(self, self.db, epoch,
                                              x_train=x_train)
                j = 0
                z += 1
                lim = np.floor(1.1**z)
            #if epoch % save_image == 0:
            #    self.do_img(x_train)
        self.do_after_train(x_train, y_train)
        self.do_save(model_save, model_name)