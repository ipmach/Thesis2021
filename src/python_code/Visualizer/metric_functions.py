import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class Metrics:

    @staticmethod
    def normalize_mse(x, y):
        return np.mean(((x - y)**2)/(np.mean((x + y) ** 2)), axis=1)

    @staticmethod
    def mse(x, y):
        return np.mean((x - y)**2, axis=1)

    @staticmethod
    def cosine_similarity(xlist, ylist):
        aux = []
        for x, y in zip(np.array(xlist), np.array(ylist)):
            aux.append(cosine_similarity(np.array([x]),
                                         np.array([y]))[0][0])
        return np.array(aux)

    @staticmethod
    def euclidian_similarity(xlist, ylist):
        aux = []
        for x, y in zip(np.array(xlist), np.array(ylist)):
            aux.append(euclidean_distances(np.array([x]),
                                           np.array([y]))[0][0])
        return np.array(aux)

    @staticmethod
    def structural_similarity(xlist, ylist):
        aux = []
        for x, y in zip(np.array(xlist), np.array(ylist)):
            aux.append(structural_similarity(np.array(x),
                                    np.array(y)))
        return np.array(aux)

    @staticmethod
    def PSNR(x, y, B=8):
        mse_ = Metrics.mse(x, y)
        R = ((2 ** B) - 1) ** 2
        aux = []
        for y in mse_:
            if y == 0:
                y = 1e-11
            aux.append(10 * np.log10(R / y))
        return np.array(aux)
