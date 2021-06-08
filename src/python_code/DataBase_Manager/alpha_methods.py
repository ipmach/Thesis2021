from DataBase_Manager.DB_technologies.tensorboard \
    import tensorboard_op as db
from DataBase_Manager.DB_technologies.numpy_ood \
    import Numpy_ood as nood


class AlphaMethods:

    @staticmethod
    def create_writer(path, name):
        """
        Create a writer
        :param path: path where to save it
        :param name: name model
        :return: the writer
        """
        return db.create_writer(path, name)

    @staticmethod
    def write_scalar(writer, names, results, epoch):
        """
        Write scalar values in a writer
        :param writer: writer where to put the scalars
        :param names: list of graphs
        :param results: list of results
        :param epoch: actual epoch
        :return:
        """
        db.write_scalar(writer, names, results, epoch)

    @staticmethod
    def write_images(writer, names, images, epoch):
        """
        Write scalar values in a writer
        :param writer: writer where to put the scalars
        :param names: list of graphs
        :param results: list of results
        :param epoch: actual epoch
        :return:
        """
        db.write_images(writer, names, images, epoch)

    @staticmethod
    def create_graph():
        """
        Initialize a graph
        :return:
        """
        db.create_graph()

    @staticmethod
    def update_graph(writer, name, path, epoch, model, inputs):
        """
        Update graph of the model
        :param writer: writer where to put the model
        :param names: list of names of the models
        :param path: path where to save
        :param epoch: actual epoch
        :return:
        """
        db.update_graph(writer, name, path, epoch, model, inputs)

    @staticmethod
    def start_performance_test(path):
        """
        Start test performance
        :param path: path where to save the performance
        :return:
        """
        db.start_performance_test(path)

    @staticmethod
    def stop_performance_test():
        """
        Stop test performance
        :return:
        """
        db.stop_performance_test()

    @staticmethod
    def create_ood_folder(path, model_name, latent_space, hidden,
                           list_datasets):
        """
        Create path folder to save data
        :param path: path where to save
        :param model_name: name of the model
        :param latent_space: size latent space
        :param hidden: size hidden space
        :param list_datasets: list of datasets used
        :return: path
        """
        return nood.create_folder_path(path, model_name, latent_space,
                                        hidden, list_datasets)

    @staticmethod
    def save_ood_data(path, dataset, x, epoch, num=None):
        """
        Save file for epoch data
        :param path: path where to save it
        :param dataset: dataset used
        :param x: data
        :param epoch: epoch num
        :return:
        """
        nood.save_data(path, dataset, x, epoch, num=num)

    @staticmethod
    def compress_path(name, path):
        """
        Save file for epoch data
        :param path: path where to save it
        :param dataset: dataset used
        :param x: data
        :param epoch: epoch num
        :return:
        """
        nood.compress_path(name, path)