import numpy as np
from sklearn.decomposition import PCA

class AutoEncoder_AAE:

    def __init__(self, model, latent_dim):
        self.model = model
        self.original_dim = self.model.image_size
        self.latent_dim = latent_dim
        self.preprocessing = self.model.preprocessing
        self.postprocessing = self.model.postprocessing
        self.preformat = self.model.preformat
        self.postformat = self.model.postformat
        self.reduce = False

    def do_encoder(self, x):
        z = np.array(self.model.encode(x))
        if self.reduce:
            z = self.pca.transform(x)
        return z

    def fit_reduce(self, x):
        z = self.do_encoder(x)
        self.pca = PCA(n_components=2).fit(z)
        self.reduce = True

    def do_decoder(self, z):
        if self.reduce:
            z = self.pca.inverse_transform(z)
        return np.array(self.model.decode(z))