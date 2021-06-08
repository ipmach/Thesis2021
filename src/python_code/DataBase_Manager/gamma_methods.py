from DataBase_Manager.DB_technologies.json_tree import JsonTree


class GammaMethods:

    @staticmethod
    def save_memory(path, data):
        """
        Save in memory the tree
        :param path: path where to save it
        :param data: tree
        :return:
        """
        JsonTree.save_memory(path, data)

    @staticmethod
    def load_memory(path):
        """
        Load a tree structure
        :param path: ath where to load it
        :return:
        """
        JsonTree.load_memory(path)

    @staticmethod
    def initialize_tree():
        """
        Initialize tree
        :return: return tree
        """
        return JsonTree.initialize_tree()

    @staticmethod
    def add_tree(tree1, tree2, name):
        """
        Add tree structure in other tree structure
        :param tree1: tree father
        :param tree2: tree child
        :param name: name tree 2
        :return: return tree 1
        """
        return JsonTree.add_tree(tree1, tree2, name)

    @staticmethod
    def copy_tree(tree):
        """
        Make a copy of a tree
        :param tree: tree structure
        :return: copy of the tree
        """
        return JsonTree.copy_tree(tree)

    @staticmethod
    def remove_branch(tree, name):
        """
        Remove branch from tree
        :param tree: tree structure
        :param name: name branch
        :return:
        """
        JsonTree.remove_branch(tree, name)

    @staticmethod
    def remove_tree(tree):
        """
        Remove all tree
        :param tree: tree structure
        :return:
        """
        JsonTree.remove_tree(tree)

    @staticmethod
    def insert_tree(tree, data, name):
        """
        Insert data in tree
        :param tree: tree structure
        :param data: data
        :param name: name to save data
        :return: tree updated
        """
        return JsonTree.insert_tree(tree, data, name)