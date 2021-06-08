from abc import ABC, abstractmethod

class GammaInterface(ABC):

    @staticmethod
    @abstractmethod
    def save_memory(path, data):
        """
        Save in memory the tree
        :param path: path where to save it
        :param data: tree
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def load_memory(path):
        """
        Load a tree structure
        :param path: ath where to load it
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def initialize_tree():
        """
        Initialize tree
        :return: return tree
        """
        pass

    @staticmethod
    @abstractmethod
    def add_tree(tree1, tree2, name):
        """
        Add tree structure in other tree structure
        :param tree1: tree father
        :param tree2: tree child
        :param name: name tree 2
        :return: return tree 1
        """
        pass

    @staticmethod
    @abstractmethod
    def copy_tree(tree):
        """
        Make a copy of a tree
        :param tree: tree structure
        :return: copy of the tree
        """
        pass

    @staticmethod
    @abstractmethod
    def remove_branch(tree, name):
        """
        Remove branch from tree
        :param tree: tree structure
        :param name: name branch
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def remove_tree(tree):
        """
        Remove all tree
        :param tree: tree structure
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def insert_tree(tree, data, name):
        """
        Insert data in tree
        :param tree: tree structure
        :param data: data
        :param name: name to save data
        :return: tree updated
        """
        pass