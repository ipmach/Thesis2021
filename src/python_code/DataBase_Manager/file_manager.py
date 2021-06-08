from DataBase_Manager.gamma_methods import GammaMethods
import string
import random
from datetime import datetime
import numpy as np
import os


class FileManager:

    def __init__(self, path, set_key=None):
        """
        Initialize file manager
        :param path: path where to save manager
        """
        # Load code_list
        self.code_list = []
        self.path = path
        try:
            self.code_list = np.load(self.path + "keys.npy")
            self.code_list = list(map(lambda x: FileManager.convert_str(x),
                           self.code_list))
        except:
            self.code_list = []

    @staticmethod
    def convert_int(index):
        """
        Convert key in ascii
        :param index: key
        :return:
        """
        return np.array([ord(c) for c in index])

    @staticmethod
    def convert_str(word):
        """
        Convert ascii key in string key
        :param word: ascii key
        :return:
        """
        return ''.join(map(chr, word))

    @staticmethod
    def generate_key(size_keys=6):
        """
        Generate a key
        :param size_keys: size of the key
        :return:
        """
        chars = string.ascii_uppercase + \
                string.ascii_lowercase + string.digits
        return ''.join(random.choice(chars) for _ in range(size_keys))

    def create_new_entrace(self, name, description,
                           time_format="%m/%d/%Y, %H:%M:%S"):
        """
        Create new entrance in the DB
        :param name: name entrance
        :param description: description of entrance
        :param time_format: time when mande
        :return: return entrance
        """
        key = FileManager.generate_key()
        while key in self.code_list:
            key = FileManager.generate_key()
        self.code_list.append(key)
        tree = GammaMethods.initialize_tree()
        tree = GammaMethods.insert_tree(tree, "key", key)
        tree = GammaMethods.insert_tree(tree, "name", name)
        tree = GammaMethods.insert_tree(tree, "description", description)
        time = datetime.now().strftime(time_format)
        tree = GammaMethods.insert_tree(tree, "date", time)
        return tree

    def insert_entrance(self, tree, name, data):
        """
        Insert data in the entrance
        :param tree: entrance
        :param name: name data
        :param data: data
        :return: entrance updated
        """
        return GammaMethods.insert_tree(tree, name , data)

    def insert_all_entrance(self, tree, names, data):
        """
        Insert more than one data
        :param tree:  entrance
        :param names: list names data
        :param data: list of data
        :return: entrance updated
        """
        for i in range(len(names)):
            tree = self.insert_entrance(tree, names[i], data[i])

    def save_entrance(self, tree):
        """
        Save entrance in memory
        :param tree: entrance
        :return:
        """
        # Save here code list
        GammaMethods.save_memory(self.path + tree['key'], tree)
        aux = list(map(lambda x: FileManager.convert_int(x),
                       self.code_list))
        np.save(self.path + "keys.npy", aux)

    def load_entrance(self, key):
        """
        Load entrance using key
        :param key: key of entrance
        :return:
        """
        return GammaMethods.load_memory(self.path + key)

    def create_set(self):
        key = FileManager.generate_key()
        while key in self.code_list:
            key = FileManager.generate_key()
        aux = list(map(lambda x: FileManager.convert_int(x),
                       self.code_list))
        #np.save(self.path + "keys.npy", aux)
        os.system("mkdir " + self.path + key)
        return key

