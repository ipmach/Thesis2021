from DataBase_Manager.alpha_methods import AlphaMethods as am


class DB_manager_train:

    def __init__(self, path, name):
        """
        :param path: path where to save data
        :param name: name model
        """
        self.path = path
        self.train_writer = am.create_writer(self.path + "/train", name)
        self.epoch = 0

    def insert_scalar_train(self, names, results, epoch=None):
        """
        Insert scalar data in train
        :param names: list of names of plot of the scalar
        :param results: list of results obtain
        :param epoch: actual epoch
        :return:
        """
        if epoch is None:
            epoch = self.epoch
        am.write_scalar(self.train_writer, names, results, epoch)
        epoch += 1
        
    def insert_images_train(self, names, images, epoch):
        """
        Insert images
        :param names: list of names of set of images
        :param images: list of set of images
        :param epoch: actual epoch
        :return:
        """
        for i in range(len(images)):
            s = images[i].shape
            try:
                images[i] = images[i].reshape((s[0], s[1], s[2], 1))
            except ValueError:
                images[i] = images[i].reshape((s[0], s[1], s[2], 3))
        am.write_images(self.train_writer, names, images, epoch)

    def start_performance(self):
        """
        Start test performance (not working yet)
        :return:
        """
        print("Start performance test")
        am.start_performance_test(self.path + "train/gradient_tape/")

    def stop_performance(self):
        """
        Stop test performance (not working yet)
        :return:
        """
        print("Stop performance test")
        am.stop_performance_test()

    def create_graph(self):
        """
        Create an empty graph
        :return:
        """
        am.create_graph()

    def upgrade_graph(self, name, epoch, model, inputs):
        """
        Upgrade graph
        :param name: name graph
        :param epoch: actual epoch
        :param model: model to plot
        :param inputs: data input for the model
        :return:
        """
        am.update_graph(self.train_writer, name,
                        self.path, epoch, model, inputs)

    def initialize_ood(self, path, model_name, latent_space, hidden,
                           list_datasets):
        """
        Initialize path folder to save data
        :param path: path where to save
        :param model_name: name of the model
        :param latent_space: size latent space
        :param hidden: size hidden space
        :param list_datasets: list of datasets used
        :return:
        """
        self.list_datasets = list_datasets
        self.ood_path = am.create_ood_folder(path, model_name,
                                             latent_space, hidden,
                                              list_datasets)

    def save_ood(self, dataset, x, epoch, num=None):
        """
        Save ood epoch data
        :param dataset: dataset used
        :param x: data
        :param epoch: epoch num
        :return:
        """
        if dataset in self.list_datasets:
            am.save_ood_data(self.ood_path, dataset, x, epoch, num=num)

    def compress(self, name):
        """
        Save ood epoch data
        :param dataset: dataset used
        :param x: data
        :param epoch: epoch num
        :return:
        """
        am.compress_path(name, self.ood_path)