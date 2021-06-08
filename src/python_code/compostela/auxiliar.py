import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri
from compostela.tree_op import Tree_op


class Auxiliar_func:

    @staticmethod
    def plot_latent_tree(z_test, y_test, cand, x_s, disc,
                         y_s, d_d_p, tree, path):
        """
        Plot in matplotlib the tree in the latent space
        :param z_test: original points in the latent space
        :param y_test: labels of original points
        :param cand: candidates
        :param x_s: points x new distribution
        :param disc: discriminator values original
        :param y_s: points y new distribution
        :param d_d_p: distributions of the candidates points
        :param tree: tree structure
        :param path: path where to save
        :return:
        """
        plt.figure("per", figsize=(14, 10))
        plt.subplot(2, 2, 1)
        triang = tri.Triangulation(x_s, y_s)
        plt.tricontour(x_s, y_s, cand, colors='k', levels=15)
        plt.tricontourf(triang, cand, levels=15)
        Tree_op.plot_tree_plt(tree)
        plt.colorbar()
        plt.legend()

        plt.subplot(2, 2, 2)
        triang = tri.Triangulation(x_s, y_s)
        plt.tricontour(x_s, y_s, cand, colors='k', levels=15)
        plt.tricontourf(triang, cand, levels=15)
        for i in range(len(d_d_p)):
            plt.scatter(d_d_p[i][:, 0], d_d_p[i][:, 1])
        Tree_op.plot_tree_plt(tree)
        plt.colorbar()

        plt.subplot(2, 2, 3)
        Tree_op.plot_tree_plt(tree)
        plt.legend()

        plt.subplot(2, 2, 4)
        triang = tri.Triangulation(z_test[:, 0], z_test[:, 1])
        plt.tricontourf(triang, disc.reshape(-1), levels=15)
        plt.scatter(z_test[:, 0],
                    z_test[:, 1], c=y_test)
        Tree_op.plot_tree_plt(tree)
        plt.savefig(path)
        plt.close()

    @staticmethod
    def create_images(model, tree, size):
        """
        Create the candidates images to .png
        :param model: Adversal Autoencoder instance
        :param tree: tree structure
        :param size: size image [height, width]
        :return: array of all images
        """
        imgs = []
        for i in tree.tree:
            imgs.append(tree[i]['point'])
        imgs = model.decode(np.array(imgs))
        return np.array(imgs).reshape((len(imgs), size[0], size[0]))

