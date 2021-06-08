from DataSets.DS.DS_interface import DS


class FashionMnist(DS):

    def __init__(self):
        super(FashionMnist, self).__init__(["Fashion"])
        self.dict = {"Fashion": self.fashion}

    def fashion(self):
        """
        Load fashion mnist dataset
        :return: x_data, y_data
        """
        return self.load("fashion_mnist")

    def __getitem__(self, item):
        """
        Get dataset
        :param item: dataset index
        :return: x_data, y_data
        """
        if item in self.variants:
            return self.dict[item]()
        return None