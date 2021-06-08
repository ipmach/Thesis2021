from abc import ABC
import tensorflow_datasets as tfds
from tqdm import tqdm
import numpy as np


class DS(ABC):

    def __init__(self, variants):
        """
        :param variants: Possible variants to load
        """
        self.variants = variants

    def load(self, name, split='train'):
        """
        Load dataset from tensorflow_datasets
        :param name: name dataset
        :return:
        """
        ds = tfds.load(name, split=split, shuffle_files=True)
        x_data = []
        y_data = []
        for i in tqdm(tfds.as_numpy(ds)):
            x_data.append(i['image'])
            y_data.append(i['label'])
        x_data = np.array(x_data).astype(float)
        y_data = np.array(y_data)
        return x_data, y_data

