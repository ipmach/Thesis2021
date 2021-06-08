import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from compostela.tree_structure import Tree_structure
from compostela.tree_op import Tree_op
from compostela.auxiliar import Auxiliar_func
from compostela.tree_optimization.kruskal import Kruskal
from termcolor import cprint


class Compostela:

    def classifier(self, model, num1, num2, z, num_labels=10, return_all=False):
        """
        Get discriminator value if a point is from one label or the other
        :param model: Adversal AutoEncoder instance
        :param num1: label 1
        :param num2: label 2
        :param point: target point
        :param num_labels: number of labels
        :return: array with [discriminator(label 1), discriminator(label 2)]
        """
        aux = tf.one_hot(tf.dtypes.cast([num1 for _ in range(len(z))],
                                        tf.int32), num_labels)
        aux2 = tf.one_hot(tf.dtypes.cast([num2 for _ in range(len(z))],
                                         tf.int32), num_labels)
        z1 = tf.concat([z, aux], -1)
        z2 = tf.concat([z, aux2], -1)
        y1 = np.array(model.discriminator(z1)).reshape(-1)
        y2 = np.array(model.discriminator(z2)).reshape(-1)
        y = (y1 - y2) ** 2
        if return_all:
            return y1, y2
        y = np.max(y) - y
        return y / np.max(y)

    def ConsineSimilarity(self, x, list_y):
        """
        Calculate cossing similary between vector x and a list of vectors
        :param x: target vector
        :param list_y: list of vectors
        :return: Cosine similarity
        """
        list_y = np.array(list_y)
        x = x.reshape((-1))
        list_y = list_y.reshape((list_y.shape[0], -1))
        return cosine_similarity([x], list_y)[0]

    def analize_point(self, model, point, s, z_f, p=0.97):
        """
        Get data of a expecific point
            sim: similarity with the rest of the sample s
            sim_d: similarity in a discrete space
            cand: set of possible candidates
            g_: population of the choosen point
        :param model: Adversal AutoEncoder instance
        :param point: Target point
        :param s: Sample
        :param z_f: discrimator values of sample s
        :param p: threeshold of candidate similarity
        :return: sim, sim_d, cand, g_
        """
        img = np.array(model.decode(np.array([point])))
        img = self.filt(img.reshape((img.shape[0], 28, 28)))
        # Calculate similirtud with the rest of the sample s
        sim = self.ConsineSimilarity(img, s)
        # Create the similarity in a discrete space (closing the area)
        # Kernel k of discrete space
        k = list(map(lambda x: 1 if x >= 0.8 else 0, z_f))
        sim_d = k * sim
        # Objective function of possible new candidates
        cand = k * (1 - (abs(sim - p)))
        # Population of the choosen point
        g = np.vectorize(lambda x: 0 if x < 0.95 else 1)
        g_ = g(sim * k)
        return sim, sim_d, cand, g_

    def define_function_population(self, z, g_):
        """
        Define a gaussian from a set of points
        :param z: latent space
        :param g_: points to use
        :return: average and standard desviation
        """
        d = z[np.nonzero(g_)[0]]
        mu = np.mean(d, axis=0)
        std = np.std(d, axis=0)
        return mu, std

    def choose_candidate(self, x_s, y_s, k_sim):
        """
        Choose a candidate with higher probability
        :param x_s: x points
        :param y_s: y points
        :param k_sim: similarity in a discrete space
        :return: candidate and probability
        """
        a = np.argmax(k_sim)
        return [x_s[a], y_s[a]], k_sim[a]

    filt = np.vectorize(lambda x: 0 if x < 0.3 else 1)

    def __call__(self, model, x_train, y_train, num1, num2, path,
                 size=[28, 28], extra=0):
        """
        Run algorithm
        Warning: Need to have a path folder where inside are img/,
                tree_performance/ json/
        :param model: Adversal Autoencoder instance
        :param x_train: original data
        :param y_train: original label
        :param num1: label 1 where num1 < num2
        :param num2: label 2 where num1 < num2
        :param path: path where to save everything
        :param size: size images
        :param extra: increase sampling area
        :return:
        """
        ## Load latent space
        z_train = np.array(model.encode(x_train))
        ## Calculate discriminator in latent space
        threshold = -0.5
        labels = tf.one_hot(tf.dtypes.cast(y_train, tf.int32), 10)
        z = tf.concat([z_train, labels], -1)
        disc = np.array(model.discriminator(z)).reshape(-1)
        ## Create random distribution
        cprint("Creating new distribution", 'blue')
        x_s = np.random.uniform(np.min(z_train[:, 0]) - extra,
                                np.max(z_train[:, 0]) + extra, 1000)
        y_s = np.random.uniform(np.min(z_train[:, 1]) - extra,
                                np.max(z_train[:, 1]) + extra, 1000)
        z_f = self.classifier(model, num1, num2,
                              np.array(list(zip(x_s, y_s))))
        z = np.array(list(zip(x_s, y_s)))
        s = np.array(model.decode(z))
        s = self.filt(s.reshape((s.shape[0], 28, 28)))
        ## Obtain initial point
        aux = np.array(z_train[np.nonzero(disc < threshold)[0]])
        initial_point = np.mean(aux, axis=0)
        cprint("Intial point " + str(initial_point), 'blue')
        ## Initialize tree
        tree = Tree_structure([num1, num2])
        sim, sim_d, cand, g_ = self.analize_point(model, initial_point,
                                                  s, z_f)
        mu, std = self.define_function_population(z, g_)
        y1, y2 = self.classifier(model, num1, num2, [initial_point],
                                 return_all=True)
        tree[0] = {"point": initial_point, "mu": mu,
                   "std": std, "father": None, "children": [],
                   "discriminator label " + str(num1): str(y1[0]),
                   "discriminator label " + str(num2): str(y2[0])}
        tree.tree_data.append(cand.copy())
        ## Rest of the tree
        cprint("Rest of the algoritm", 'blue')
        d_d_p = []
        d_d_p.append(np.array(tf.random.normal([200, 2],
                                               mean=mu, stddev=std)))
        cands = []
        father = 0
        while True:
            p = 1
            i_ = father
            while True:
                new_candidate, p = self.choose_candidate(x_s, y_s,
                                                         tree.tree_data[father] - g_)
                if p < 0.7:
                    break
                i_ += 1
                _, _, cand2, g_3 = self.analize_point(model, new_candidate,
                                                       s, z_f)
                mu_3, std_3 = self.define_function_population(z, g_3)
                d_d_p.append(np.array(tf.random.normal([200, 2],
                                                       mean=mu_3, stddev=std_3)))
                g_ += g_3
                y1, y2 = self.classifier(model, num1, num2, [new_candidate],
                                                return_all=True)
                tree[i_] = {"point": new_candidate.copy(), "mu": mu_3.copy(),
                             "std": std_3.copy(), "father": father, "children": [],
                            "discriminator label " + str(num1): str(y1[0]),
                            "discriminator label " + str(num2): str(y2[0])}
                tree[father]["children"].append(i_)
                cands.append(cand2.copy())

            for i in range(len(tree[father]["children"])):
                tree.tree_data.append(cands[i].copy())

            father += 1
            if father >= len(tree):
                break
        ## Optimize tree
        cprint("Optimizing tree", 'blue')
        kr = Kruskal()
        tree = kr(tree)
        ## Create images
        cprint("Saving and plotting", 'blue')
        imgs = Auxiliar_func.create_images(model, tree, size)
        ## Create plots 
        #Tree_op.print_tree(0, tree)
        path_latent = path + "tree_performance/" + str(num1) + "-" + str(num2)
        Auxiliar_func.plot_latent_tree(z_train, y_train, cand, x_s, disc,
                                       y_s, d_d_p, tree, path_latent)
        path_candidates = path + "imgs/" + str(num1) + "-" + str(num2) + "/"
        path_tree = path + "tree/" + str(num1) + "-" + str(num2)
        Tree_op.plot_tree(tree, imgs, num1, num2, path=path_candidates,
                          path_tree=path_tree)
        tree.save(path + "json/")

        #Save in numpy
        points = np.array([[float(tree[str(t)]['point'][0]),
                            float(tree[str(t)]['point'][1])]
                           for t in range(len(tree))])
        np.save(path + "numpy/data" + str(num1) + "_" + str(num2) + ".npy",
                np.array(model.decode(points)))
