#!/usr/bin/env python

from Visualizer.validation import Validation
from Models.model_VAE import VAE
from Visualizer.distribution_wrapper import Distribution
from Visualizer.wrapper_AE import AutoEncoder_AAE
import tensorflow as tf
import numpy as np
from Models.model_AAE import AAE


hidden_dim = 64
latent_dim = 2
original_dim = 28 * 28
path = "Outputs/hHI1rw/VAE_models/"
#path = "Outputs/NG1PzB/VAE_models/"

value1 = 0
value2 = 2
"""
path_model = "Results_Experiments/interpolation_training/1/Models/"
encoder_path = path_model + "model_AAE_" + str(value1) + "_" + str(value2)+ "_encoder$z_2$h_64.h5"
decoder_path = path_model + "model_AAE_" + str(value1) + "_" + str(value2)+ "_decoder$z_2$h_64.h5"
discriminator_path = path_model + "model_AAE_" + str(value1) + "_" + str(value2)+ "_discriminator$z_2$h_64.h5"
model = AAE([hidden_dim, latent_dim, original_dim], 10)
model.load_weights_model([encoder_path, decoder_path,
                          discriminator_path])
"""
model = VAE([hidden_dim, latent_dim, original_dim])
model.build(input_shape=(4, original_dim))
model.load_weights(path + "model_32_all.h5")


model_ = AutoEncoder_AAE(model, 2)

models = []
for i in range(10):
    models.append(VAE([hidden_dim, latent_dim, original_dim]))
    models[-1].build(input_shape=(4, original_dim))
    models[-1].load_weights(path + "model_32_" + str(i) + ".h5")
    models[-1] = AutoEncoder_AAE(models[-1], 2)

train, test = tf.keras.datasets.mnist.load_data()
(x_train, y_train) = train
#####
"""
train_index_0 = np.array(list(map(lambda x: (x==value1) , y_train)))
train_index_0 = train_index_0.astype(int)
train_index_1 = np.array(list(map(lambda x: (x == value2), y_train)))
train_index_1 = train_index_1.astype(int)
train_index = train_index_0 + train_index_1
train_index = np.nonzero(train_index)[0]
y_train_ = y_train[train_index]
x_train_ = x_train[train_index]
"""
#####
name_labels = ["Label 0", "Label 1", "Label 2", "Label 3", "Label 4",
               "Label 5", "Label 6", "Label 7", "Label 8", "Label 9"]


"""
x_train = model.preformat(x_train)
x_train = model.preprocessing(x_train)

x_train_ = model.preformat(x_train_)
x_train_ = model.preprocessing(x_train_)
z_train = np.array(model.encode(x_train_))
D = Distribution(z_train[:,0], z_train[:,1], y_train_, name_labels)
"""

x_train = model.preformat(x_train)
x_train = model.preprocessing(x_train)
z_train = np.array(model.encode(x_train))
D = Distribution(z_train[:,0], z_train[:,1], y_train, name_labels)

train_d_x = []
for i in range(10):
    train_index = np.array(list(map(lambda x: (x==i), y_train)))
    train_index = train_index.astype(int)
    train_index = np.nonzero(train_index)[0]
    train_d_x.append(x_train[train_index])


D.original = train_d_x


a = Validation(model_, models, D)
a.show()
