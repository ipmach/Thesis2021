from DataSets.DS.DS_interface import DS
import numpy as np
import json


class MNIST(DS):

    def __init__(self):
        super(MNIST, self).__init__(["MNIST", "MNIST-C"])
        self.dict = {"MNIST": self.mnist, "MNIST-C": self.mnist_c}

    def mnist(self):
        """
        Load mnist dataset
        :return: x_data, y_data
        """
        return self.load("MNIST")

    def mnist_c(self, size_each=4000, save=True,
                path_json='src/python_code/settings.json'):
        """
        Load corrupt mnist dataset
        :param size_each: size of each corruption
        :param save: save data in cache files
        :param path_json: settings file
        :return: x_data, y_data
        """
        settings = json.load(open(path_json))["Datasets"]["MNIST-C"]
        path_file_x = settings["x_file"]
        path_file_y = settings["y_file"]
        try:
            x_data = np.load(path_file_x + ".npy")
            y_data = np.load(path_file_y + ".npy")
            return x_data, y_data
        except Exception:
            corruptions = ["mnist_corrupted/shot_noise",
                           "mnist_corrupted/impulse_noise",
                           "mnist_corrupted/glass_blur",
                           "mnist_corrupted/motion_blur",
                           "mnist_corrupted/shear",
                           "mnist_corrupted/scale",
                           "mnist_corrupted/rotate",
                           "mnist_corrupted/brightness",
                           "mnist_corrupted/translate",
                           "mnist_corrupted/stripe",
                           "mnist_corrupted/fog",
                           "mnist_corrupted/spatter",
                           "mnist_corrupted/dotted_line",
                           "mnist_corrupted/zigzag",
                           "mnist_corrupted/canny_edges"]

            x_data, y_data = self.load(corruptions[0])
            x_data = x_data[:size_each]
            y_data = y_data[:size_each]
            corruptions = corruptions[1:]
            for i in corruptions:
                x_data2, y_data2 = self.load(i)
                x_data2 = x_data2[:size_each]
                y_data2 = y_data2[:size_each]
                x_data = np.append(x_data, x_data2, axis=0)
                y_data = np.append(y_data, y_data2, axis=0)
            index = np.arange(y_data.shape[0])
            np.random.shuffle(index)
            x_data = x_data[index]
            y_data = y_data[index]
            if save:
                np.save(path_file_x + ".npy", x_data)
                np.save(path_file_y + ".npy", y_data)
            return x_data, y_data

    def __getitem__(self, item):
        """
        Get dataset
        :param item: dataset index
        :return: x_data, y_data
        """
        if item in self.variants:
            return self.dict[item]()
        return None




