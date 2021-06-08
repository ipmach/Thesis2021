from abc import ABC, abstractmethod


class alpha_ood_interface(ABC):

    @staticmethod
    @abstractmethod
    def save_data(path, x, epoch, num=None):
        """
        Save numpy file epoch data
        :param path: path where to save it
        :param x: data
        :param epoch: epoch num
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def create_folder_path(path, model_name, latent_space, hidden,
                           list_datasets):
        """
        Create path folder to save data
        :param path: path where to save
        :param model_name: name of the model
        :param latent_space: size latent space
        :param hidden: size hidden space
        :param list_datasets: list of datasets used
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def compress_path(name, path):
        """
        Compress folder
        :param name: name zip
        :param path: path
        :return:
        """
        pass