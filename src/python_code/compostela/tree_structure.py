import json


class Tree_structure:
    """
    Tree structure use in the algorithm
        tree: tree structure
        tree_data: external data need it
    """

    def __init__(self, nums):
        self.tree = {}
        self.tree_data = []
        self.num1 = nums[0]
        self.num2 = nums[1]

    def __getitem__(self, item):
        """
        Get element of the tree like an array or tree
        :param item: index of element (can be a int or a str)
        :return: The elemt
        """
        item = str(item)
        return self.tree[item]

    def __setitem__(self, item, value):
        """
        Set element of the tree with a value
        :param item: index of the element (can be a int or a str)
        :param value: value of the element
        :return:
        """
        item = str(item)
        self.tree[item] = value

    def __len__(self):
        return len(self.tree)

    def save(self, path):
        """
        Save the tree in a json file
        :param path: path where to save the json file
        :return:
        """
        aux = {}
        # We need to convert floats into string for JSON
        for j, t in enumerate(self.tree):
            aux[j] = self.tree[t]
            aux[j]['point'] = list([str(i) for i in aux[j]['point']])
            aux[j]['mu'] = list([str(i) for i in aux[j]['mu']])
            aux[j]['std'] = list([str(i) for i in aux[j]['std']])

        with open(path + str(self.num1) + "-" + \
                  str(self.num2) + '.json', 'w') as f:
            json.dump(aux, f)