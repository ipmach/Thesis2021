from DataBase_Manager.DB_technologies.alpha_ood_interface import alpha_ood_interface
import numpy as np
import os


class Numpy_ood(alpha_ood_interface):

    @staticmethod
    def save_data(path, dataset, x, epoch, num=None):
        """
        Save numpy file epoch data
        :param path: path where to save it
        :param dataset: dataset used
        :param x: data
        :param epoch: epoch num
        :return:
        """
        if num is not None:
            np.save(path + "/" + dataset + "/" + "Epoch_" + str(epoch) + " " + str(num) + ".npy", x)
        else:
            np.save(path + "/" + dataset + "/" + "Epoch_" + str(epoch) + ".npy", x)

    @staticmethod
    def create_folder_path(path, model_name, latent_space, hidden,
                           list_datasets):
        """
        Create path folder to save data
        :param path: path where to save
        :param model_name: name of the model
        :param latent_space: size latent space
        :param hidden: size hidden space
        :param list_datasets: list of datasets used
        :return: path
        """
        try:
            path = path + "/" + model_name
            os.mkdir(path)
        except FileExistsError:
            pass
        try:
            path = path + "/latent_space_" + str(latent_space)
            os.mkdir(path)
        except FileExistsError:
            pass
        try:
            path = path + "/hidden_space_" + str(hidden)
            os.mkdir(path)
        except FileExistsError:
            pass
        for dataset in list_datasets:
            os.mkdir(path + "/" + dataset)
        return path
    
    @staticmethod
    def compress_path(name, path):
        """
        Compress folder
        :param name: name zip
        :param path: path
        :return:
        """
        os.system("zip -r " + name + " " + path)
        #os.system("rm -r " + path)