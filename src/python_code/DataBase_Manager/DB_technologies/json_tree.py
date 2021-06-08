from DataBase_Manager.DB_technologies.gamma_interface import GammaInterface
import json
import copy


class JsonTree(GammaInterface):

    @staticmethod
    def save_memory(path, data):
        """
        Save in memory the tree
        :param path: path where to save it
        :param data: tree
        :return:
        """
        with open(path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load_memory(path):
        """
        Load a tree structure
        :param path: ath where to load it
        :return:
        """
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def initialize_tree():
        """
        Initialize tree
        :return: return tree
        """
        return {}

    @staticmethod
    def add_tree(tree1, tree2, name):
        """
        Add tree structure in other tree structure
        :param tree1: tree father
        :param tree2: tree child
        :param name: name tree 2
        :return: return tree 1
        """
        tree1[name] = tree2
        return tree1

    @staticmethod
    def copy_tree(tree):
        """
        Make a copy of a tree
        :param tree: tree structure
        :return: copy of the tree
        """
        return copy.deepcopy(tree)

    @staticmethod
    def remove_branch(tree, name):
        """
        Remove branch from tree
        :param tree: tree structure
        :param name: name branch
        :return:
        """
        del tree[name]

    @staticmethod
    def remove_tree(tree):
        """
        Remove all tree
        :param tree: tree structure
        :return:
        """
        del tree

    @staticmethod
    def insert_tree(tree, data, name):
        """
        Insert data in tree
        :param tree: tree structure
        :param data: data
        :param name: name to save data
        :return: tree updated
        """
        tree[data] = name
        return tree