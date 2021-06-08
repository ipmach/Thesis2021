import tensorflow as tf
import datetime
from DataBase_Manager.DB_technologies.alpha_interface import AlphaInterface


class tensorboard_op(AlphaInterface):

    @staticmethod
    def create_writer(path, name):
        """
        Create a writer in the tensorboard
        :param path: path where to save it
        :param name: name model
        :return: the writer
        """
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = path + '/gradient_tape/' + current_time + "-" + name
        return tf.summary.create_file_writer(train_log_dir)

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
        with writer.as_default():
            for i in range(len(names)):
                tf.summary.scalar(names[i], results[i], step=epoch)

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
        with writer.as_default():
            for i in range(len(names)):
                tf.summary.image(names[i], images[i], step=epoch)

    @staticmethod
    def create_graph():
        """
        Initialize a graph
        :return:
        """
        tf.summary.trace_on(graph=True, profiler=True)

    @staticmethod
    def update_graph(writer, name, path, epoch, model, inputs):
        """
        Update graph of the model
        :param writer: writer where to put the model
        :param names: list of names of the models
        :param path: path where to save
        :param epoch: actual epoch
        :param model: model to graph
        :param inputs: input of the model
        :return:
        """
        @tf.function
        def traceme(x):
            return model(x)
        traceme(inputs)
        with writer.as_default():
            tf.summary.trace_export(
                name=name,
                step=epoch,
                profiler_outdir=path)

    @staticmethod
    def start_performance_test(path):
        """
        Start test performance
        :param path: path where to save the performance
        :return:
        """
        tf.profiler.experimental.start(path)

    @staticmethod
    def stop_performance_test():
        """
        Stop test performance
        :return:
        """
        tf.profiler.experimental.stop()


