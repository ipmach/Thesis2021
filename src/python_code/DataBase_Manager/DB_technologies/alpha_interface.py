from abc import ABC, abstractmethod


class AlphaInterface(ABC):

    @staticmethod
    @abstractmethod
    def create_writer(path, name):
        """
        Create a writer
        :param path: path where to save it
        :param name: name model
        :return: the writer
        """
        pass


    @staticmethod
    @abstractmethod
    def write_scalar(writer, names, results, epoch):
        """
        Write scalar values in a writer
        :param writer: writer where to put the scalars
        :param names: list of graphs
        :param results: list of results
        :param epoch: actual epoch
        :return:
        """
        pass


    @staticmethod
    @abstractmethod
    def write_images(writer, names, images, epoch):
        """
        Write scalar values in a writer
        :param writer: writer where to put the scalars
        :param names: list of graphs
        :param results: list of results
        :param epoch: actual epoch
        :return:
        """
        pass


    @staticmethod
    @abstractmethod
    def create_graph():
        """
        Initialize a graph
        :return:
        """
        pass


    @staticmethod
    @abstractmethod
    def update_graph(writer, name, path, epoch, model, inputs):
        """
        Update graph of the model
        :param writer: writer where to put the model
        :param name: list of names of the models
        :param path: path where to save
        :param epoch: actual epoch
        :param model: model to graph
        :param inputs: input of the model
        :return:
        """
        pass


    @staticmethod
    @abstractmethod
    def start_performance_test(path):
        """
        Start test performance
        :param path: path where to save the performance
        :return:
        """
        pass


    @staticmethod
    @abstractmethod
    def stop_performance_test():
        """
        Stop test performance
        :return:
        """
        pass