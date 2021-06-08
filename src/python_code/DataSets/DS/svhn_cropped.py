from DataSets.DS.DS_interface import DS
import numpy as np
import json


class SVHN(DS):

    def __init__(self, path_json = 'src/python_code/settings.json'):
        super(SVHN, self).__init__(["svhn_cropped"])
        self.dict = {"svhn_cropped": self.svhn}
        self.apply_mean = bool(int(json.load(open(path_json))["Datasets"]["SVHN"]["apply_gray"]))
        self.path_json = path_json

    def svhn(self,  save=True):
        """
        Load cifar10 dataset
        :return: x_data, y_data
        :param path_json: settings file
        """
        settings = json.load(open(self.path_json))["Datasets"]["SVHN"]
        path_file_x = settings["x_file"]
        path_file_y = settings["y_file"]
        type_ = "_gray" if self.apply_mean else "_color"
        try:
            x_data = np.load(path_file_x + type_ + ".npy")
            y_data = np.load(path_file_y + type_ + ".npy")
            return x_data, y_data
        except Exception:
            x_data, y_data = self.load("svhn_cropped")
            if self.apply_mean:
                x_data = np.mean(x_data, axis=3)
            if save:
                np.save(path_file_x + type_ + ".npy", x_data)
                np.save(path_file_y + type_ + ".npy", y_data)
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