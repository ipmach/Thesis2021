from abc import ABC, abstractmethod
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import json


class TrainInterface(ABC):

    def init_train(self, num_epochs, batch_size,
                 learning_rate, db):
        """
         Interface train
        :param num_epochs: number of epochs in training
        :param batch_size: Bacth size in training
        :param learning_rate: Learning rate
        :param db: Db where to save data training
        :param kwargs: optional parameters for VAE
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.db = db
        self.disc = False
        self.likelihood = False
        self.svm1 = False
        self.loglikelihood = False
        self.mahalanobis = False


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
                ood_op.operate_data(self, self.db, epoch,
                                    x_train=x_train, disc=self.disc,
                                    likelihood=self.likelihood, svm1=self.svm1,
                                    loglikelihood=self.loglikelihood, mahalanobis=self.mahalanobis)
                j = 0
                z += 1
                lim = np.floor(1.1**z)
            if epoch % save_image == 0:
                self.do_img(x_train, epoch)
        self.do_after_train(x_train, y_train)
        self.do_save(model_save, model_name)

    def init_dataset(self, x_train, y_train):
        """
        Initialize dataset
        :param x_train:
        :param y_train:
        :return:
        """
        dataset = tf.data.Dataset.from_tensor_slices(x_train)
        return dataset.shuffle(self.batch_size * 5).batch(self.batch_size)

    def pre_model(self, x):
        """
        Class that process the preprocess part
        x: data
        :return:
        """
        x_ = self.preprocessing(x)
        return self.preformat(x_)

    def post_model(self, x):
        """
        Class that process the postprocess part
        x: data
        :return:
        """
        x_ = self.postprocessing(x)
        return self.postformat(x_)

    @abstractmethod
    def do_epoch(self, dataset):
        """
        Do operation epoch
        :param dataset:
        :return:
        """
        pass

    def do_img(self, x_train, epoch):
        """
        Save images
        :param x_train: all data
        :return:
        """
        x = self.postformat(np.array(x_train))
        x = x_train[:5]
        x_ = self(x)
        #x_ = self.postprocessing(x_)
        x_ = self.postformat(np.array(x_))
        self.db.insert_images_train(["Reconstructions " + self.model_name + " " + str(epoch)],
                                    [x_], epoch)
        if epoch == 0:
            #x = self.postprocessing(x)
            x = self.postformat(np.array(x))
            self.db.insert_images_train(["Original " + self.model_name],
                                        [x], epoch)


    @abstractmethod
    def do_save(self, model_save, model_name):
        """
        Save model
        :return:
        """
        pass

    def do_before_train(self, x_train, y_train):
        """
        Do before first epoch
        :param y_train:
        :return:
        """
        return None

    def do_after_train(self, x_train, y_train):
        """
        Do after last epoch
        :param y_train:
        :return:
        """
        return None



