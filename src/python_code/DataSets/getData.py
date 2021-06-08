from DataSets.DS.fashion_mnist import FashionMnist
from DataSets.DS.svhn_cropped import SVHN
from DataSets.DS.cifar10 import Cifar10
from DataSets.DS.mnist import MNIST
import numpy as np


class GetData:

    @staticmethod
    def get_ds(name):
        """
        Get Dataset from DS folder
        :param name: Name dataset
        :return: x_data, y_data
        """
        if name == "MNIST":
            ds = MNIST()
            return ds[name]
        elif name == "MNIST-C":
            ds = MNIST()
            return ds[name]
        elif name == "FashionMnist":
            ds = FashionMnist()
            return ds["Fashion"]
        elif name == "svhn_cropped":
            ds = SVHN()
            return ds["svhn_cropped"]
        elif name == "cifar10":
            ds = Cifar10()
            return ds["cifar10"]
        elif name == "cifar10-C":
            ds = Cifar10()
            return ds["cifar10-C"]
        else:
            print("Not found")
            return None

    @staticmethod
    def get_filter(x_data, y_data, list_values):
        """
        Filter data
        :param x_data: original x_data
        :param y_data: original y_data
        :param list_values: list values to take
        :return: new x_data, new y_data
        """
        for j, value in enumerate(list_values):
            if j == 0:
                x_data_index = np.array(list(map(lambda x: (x == value), y_data)))
                x_data_index = x_data_index.astype(int)
            else:
                x_data_index2 = np.array(list(map(lambda x: (x == value), y_data)))
                x_data_index2= x_data_index2.astype(int)
                x_data_index = x_data_index + x_data_index2
        x_data_index = np.nonzero(x_data_index)[0]
        x_data = x_data[x_data_index]
        y_data = y_data[x_data_index]
        return x_data, y_data

    @staticmethod
    def split_data(x_data, y_data, split=0.8):
        """
        Split data
        :param x_data: original x_data
        :param y_data: original y_data
        :param split: percentage to split [0,1]
        :return: x_split_1, y_split_1, x_split_2, y_split_2
        """
        split = int(len(x_data) * split)
        x_split_1 = x_data[:split]
        y_split_1 = y_data[:split]
        x_split_2 = x_data[split:]
        y_split_2 = y_data[split:]
        return x_split_1, y_split_1, x_split_2, y_split_2


