from DataSets.DS.DS_interface import DS
import numpy as np
import json


class Cifar10(DS):

    def __init__(self, path_json='src/python_code/settings.json'):
        super(Cifar10, self).__init__(["cifar10", "cifar10-C"])
        self.dict = {"cifar10": self.cifar10, "cifar10-C": self.cifar10_c}
        self.apply_mean = bool(int(json.load(open(path_json))["Datasets"]["CIFAR10"]["apply_gray"]))
        self.path_json = path_json

    def cifar10(self):
        """
        Load cifar10 dataset
        :return: x_data, y_data
        """
        x_data, y_data = self.load("cifar10")
        if self.apply_mean:
            x_data = np.mean(x_data, axis=3)
        return x_data, y_data

    def cifar10_c(self, size_each=3157, save=True, level=5):
        """
        Load corrupt mnist dataset
        :param size_each: size of each corruption
        :param save: save data in cache files
        :param level: level of corruptnes
        :return: x_data, y_data
        """
        settings = json.load(open(self.path_json))["Datasets"]["CIFAR10-C"]
        path_file_x = settings["x_file"]
        path_file_y = settings["y_file"]
        type_ = "_gray" if self.apply_mean else "_color"
        try:
            x_data = np.load(path_file_x + str(level) + type_ + ".npy")
            y_data = np.load(path_file_y + str(level) + type_ + ".npy")
            return x_data, y_data
        except Exception:
            corruptions = ["cifar10_corrupted/brightness_" + str(level),
                           "cifar10_corrupted/contrast_" + str(level),
                           "cifar10_corrupted/defocus_blur_" + str(level),
                           "cifar10_corrupted/elastic_" + str(level),
                           "cifar10_corrupted/fog_" + str(level),
                           "cifar10_corrupted/frost_" + str(level),
                           "cifar10_corrupted/frosted_glass_blur_" + str(level),
                           "cifar10_corrupted/gaussian_blur_" + str(level),
                           "cifar10_corrupted/gaussian_noise_" + str(level),
                           "cifar10_corrupted/impulse_noise_" + str(level),
                           "cifar10_corrupted/jpeg_compression_" + str(level),
                           "cifar10_corrupted/motion_blur_" + str(level),
                           "cifar10_corrupted/pixelate_" + str(level),
                           "cifar10_corrupted/saturate_" + str(level),
                           "cifar10_corrupted/shot_noise_" + str(level),
                           "cifar10_corrupted/snow_" + str(level),
                           "cifar10_corrupted/spatter_" + str(level),
                           "cifar10_corrupted/speckle_noise_" + str(level),
                           "cifar10_corrupted/zoom_blur_" + str(level)]
            x_data, y_data = self.load(corruptions[0], split='test')
            x_data = x_data[:size_each]
            y_data = y_data[:size_each]
            corruptions = corruptions[1:]
            for i in corruptions:
                x_data2, y_data2 = self.load(i, split='test')
                x_data2 = x_data2[:size_each]
                y_data2 = y_data2[:size_each]
                x_data = np.append(x_data, x_data2, axis=0)
                y_data = np.append(y_data, y_data2, axis=0)
            x_data = x_data[:60000]
            y_data = y_data[:60000]
            index = np.arange(y_data.shape[0])
            np.random.shuffle(index)
            x_data = x_data[index]
            y_data = y_data[index]
            if self.apply_mean:
                x_data = np.mean(x_data, axis=3)
            if save:
                np.save(path_file_x + str(level) + type_ + ".npy", x_data)
                np.save(path_file_y + str(level) + type_ + ".npy", y_data)
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