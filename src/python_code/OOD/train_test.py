from OOD.metrics import Metrics, BinaryCross, Mahalanobis
import tensorflow_probability as tfp
from sklearn.svm import OneClassSVM
import tensorflow as tf
import numpy as np
import json


class OODTest:

    def __init__(self, odd_names, path_json='src/python_code/settings.json'):
        bc = BinaryCross()
        metric_use = json.load(open(path_json))["OOD"]["Gather_Data"]["Reconstruction"]
        self.metric_list = {"MSE": Metrics.MSE,
                            "NormLikehood": Metrics.NormLikehood,
                            "BinaryCross": bc,
                            "LogNormLikelihood": Metrics.Log_NormLikehood}
        self.metric = self.metric_list[metric_use]
        self.metric_use = metric_use
        self.odd_names = [name + " " + metric_use for name in odd_names]
        self.odd_data = [[] for _ in range(len(self.odd_names))]
        self.test = None
        settings = json.load(open('src/python_code/settings.json'))
        self.batch_size = settings['OOD']["batch_size"]
        self.test_size = int(settings['OOD']["id_test_size"])
        self.train_size = int(settings['OOD']["id_train_size"])
        self.ood_size = int(settings['OOD']["ood_size"])

    def initialize_data(self, model, ood_data, x_test=None):
        """
        Initialize data in order to operatate the ood tests
        :param model: model in the current phase
        :param ood_data: list of ood data
        :param x_test: (optional) add test data
        :return:
        """
        for j, data in enumerate(ood_data):
            aux = model.preprocessing(data[:self.ood_size])
            aux = model.preformat(aux)
            self.odd_data[j] = tf.convert_to_tensor(aux.copy())

        if x_test is not None and self.test_size > 0:
            aux = model.preprocessing(x_test[:self.test_size])
            aux = model.preformat(aux)
            self.test = tf.convert_to_tensor(aux.copy())

    def do_img(self, x, x_, epoch, name, db, model):
        """
        Save images
        :param x: all data
        :param epoch: actual epoch
        :param name: name dataset
        :param db: database use
        :param model: model use
        :return:
        """
        #print(epoch % self.save_image, epoch, self.save_image)
        x = x[:5]
        x_ = x_[:5]
        # x_ = self.postprocessing(x_)
        x_ = model.postformat(np.array(x_))
        db.insert_images_train(["OOD " + name + " " + model.model_name + " " + str(epoch)],
                                [x_], epoch)
        if epoch == 1:
            #x = model.postprocessing(x)
            x = model.postformat(np.array(x))
            db.insert_images_train(["Original OOD " + name + " " + model.model_name],
                                    [x], epoch)

    def operate_data(self, model, db, epoch, x_train=None, disc=False,
                     likelihood=False, svm1=False, loglikelihood=False, mahalanobis=False):
        """
        Test ood data in the current state of the model
        :param model: model in the current state
        :param db: database where to save results
        :param epoch: epoch of the training
        :param x_train: data from the training
        :param disc: for AAE discriminator
        :return:
        """
        results = []
        names = self.odd_names.copy()
        for j, data in enumerate(self.odd_data):  # Testing OOD data reconstruction
            data = data
            aux = model.predict(data, batch_size=self.batch_size)
            self.do_img(data, aux, epoch, names[j], db, model)
            results.append(self.metric(data, aux, db, names[j], epoch))

        if self.test is not None:  # Testing ID test data reconstruction
            aux = model.predict(self.test, batch_size=self.batch_size)
            results.append(self.metric(self.test, aux, db,
                                       "Test OOD " + self.metric_use,
                                       epoch))
            names = names + ["Test OOD " + self.metric_use]
            self.do_img(self.test   , aux, epoch, names[-1], db, model)

        if x_train is not None:  # Testing ID train data reconstruction
            aux = model.predict(x_train[:self.train_size],
                                batch_size=self.batch_size)
            results.append(self.metric(x_train[:self.train_size], aux, db,
                                       "Train OOD " + self.metric_use,
                                       epoch))
            names = names + ["Train OOD " + self.metric_use]
            self.do_img(x_train[:self.train_size], aux, epoch, names[-1], db, model)

        if disc:  # Testing feature extraction with discriminator model
            for j, data in enumerate(self.odd_data):  # Testing OOD data
                aux = model.encode_.predict(data, batch_size=self.batch_size)
                aux = model.discriminator_.predict(aux, batch_size=self.batch_size)
                db.save_ood(names[j] + " Disc", aux, epoch)

            if self.test is not None:  # Testing ID test data
                aux = model.encode_.predict(self.test, batch_size=self.batch_size)
                aux = model.discriminator_.predict(aux, batch_size=self.batch_size)
                db.save_ood("Test OOD Disc", aux, epoch)

            if x_train is not None:  # Testing ID train data
                aux = model.encode_.predict(x_train[:self.train_size], batch_size=self.batch_size)
                aux = model.discriminator_.predict(aux, batch_size=self.batch_size)
                db.save_ood("Train OOD Disc", aux, epoch)

        if likelihood:  # Testing feature extraction with likelihood
            tfd = tfp.distributions
            dist = tfd.Normal(loc=0., scale=1.)  # Normal distribution as predefine one
            for j, data in enumerate(self.odd_data):  # Testing OOD data
                aux = model.encode_.predict(data, batch_size=self.batch_size)
                self.metric_list["NormLikehood"](aux, db, names[j] + " Likehood", epoch, dist)

            if self.test is not None:  # Testing ID test data
                aux = model.encode_.predict(self.test, batch_size=self.batch_size)
                self.metric_list["NormLikehood"](aux, db, "Test OOD Likehood", epoch, dist)

            if x_train is not None:  # Testing ID train data
                aux = model.encode_.predict(x_train[:self.train_size], batch_size=self.batch_size)
                self.metric_list["NormLikehood"](aux, db, "Train OOD Likehood", epoch, dist)

        if loglikelihood:  # Testing feature extraction with log likelihood
            tfd = tfp.distributions
            dist = tfd.Normal(loc=0., scale=1.)  # Normal distribution as predefine one
            for j, data in enumerate(self.odd_data):  # Testing OOD data
                aux = model.encode_.predict(data, batch_size=self.batch_size)
                self.metric_list["LogNormLikelihood"](aux, db, names[j] + " LogNormLikelihood", epoch, dist)

            if self.test is not None:  # Testing ID test data
                aux = model.encode_.predict(self.test, batch_size=self.batch_size)
                self.metric_list["LogNormLikelihood"](aux, db, "Test OOD LogNormLikelihood", epoch, dist)

            if x_train is not None:  # Testing ID train data
                aux = model.encode_.predict(x_train[:self.train_size], batch_size=self.batch_size)
                self.metric_list["LogNormLikelihood"](aux, db, "Train OOD LogNormLikelihood", epoch, dist)

        if mahalanobis: # Testing with mahalanobis distance
            x = model.encode_.predict(x_train[:self.train_size], batch_size=self.batch_size)
            Dm = Mahalanobis(x)
            Dm(aux, db, "Train OOD Mahalanobis", epoch)
            for j, data in enumerate(self.odd_data):  # Testing OOD data
                aux = model.encode_.predict(data, batch_size=self.batch_size)
                Dm(aux, db, names[j] + " Mahalanobis", epoch)

        if svm1:  # Testing feature extraction with svm1
            #test_enc = model.encode_.predict(self.test, batch_size=self.batch_size)
            no_ood = model.encode_.predict(x_train[:self.train_size], batch_size=self.batch_size)
            #no_ood = tf.concat([test_enc, train_enc], axis=0)
            for j, data in enumerate(self.odd_data):
                aux = model.encode_.predict(data, batch_size=self.batch_size)
                X = tf.concat([no_ood, aux], axis=0)
                X = tf.reshape(X, (-1, X.shape[2]))
                clf = OneClassSVM(kernel='poly', degree=3, nu=0.5).fit(X)
                Y = clf.predict(X)
                db.save_ood(names[j] + " SVM1", Y, epoch)


        db.insert_scalar_train(names, results, epoch=epoch)

    def operate_data_ensembles(self, model, db, epoch, x_train=None):
        """
        Test ood data in the current state of the model,
        with ensembles models
        :param model: model in the current state
        :param db: database where to save results
        :param epoch: epoch of the training
        :param x_train: data from the training
        :return:
        """
        results = []
        names = self.odd_names.copy()
        for j, data in enumerate(self.odd_data):
            results_aux = []
            for i in range(len(model)):
                data = data[:60000]
                aux = model[i].predict(data, batch_size=self.batch_size)
                results_aux.append(self.metric(data, aux, db,
                                       names[j],
                                       epoch, num=i))
            results.append(np.median(results_aux))

        if self.test is not None:
            results_aux = []
            for j in range(len(model)):
                aux = model[j].predict(self.test, batch_size=self.batch_size)
                results_aux.append(self.metric(self.test, aux, db,
                                               "Test OOD " + self.metric_use,
                                               epoch, num=j))
            results.append(np.median(results_aux))

        if x_train is not None:
            results_aux = []
            for j in range(len(model)):
                aux = model[j].predict(x_train[:self.train_size],
                                       batch_size=self.batch_size)
                results_aux.append(self.metric(x_train[:self.train_size], aux, db,
                                               "Train OOD " + self.metric_use,
                                               epoch, num=j))
            results.append(np.median(results_aux))
        db.insert_scalar_train(names, results, epoch=epoch)




