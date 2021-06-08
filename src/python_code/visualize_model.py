#!/usr/bin/env python

import numpy as np
from Models.model_AAE import AAE
from Models.model_VAE import VAE
from Models.model_PAE import PAE
from Visualizer.distribution_wrapper import Distribution
from Visualizer.Figure import Figure
from Visualizer.wrapper_AE import AutoEncoder_AAE
import tensorflow as tf
import json
import sys

# Hyper parameters
original_dim = int(sys.argv[1])
latent_dim = int(sys.argv[2])
hidden_dim = int(sys.argv[3])
encoder_path = sys.argv[4]
decoder_path = sys.argv[5]
path_data = sys.argv[6]
type_encoder = sys.argv[7]

path_tree = None
# Load model
if type_encoder == "AAE":
    discriminator_path = sys.argv[8]
    if sys.argv[9] != "None":
        path_tree = sys.argv[9]
    model = AAE([hidden_dim, latent_dim, original_dim], 10)
    model.load_weights_model([encoder_path, decoder_path,
                              discriminator_path])
elif type_encoder == "VAE":
    model = VAE([hidden_dim, latent_dim, original_dim])
    model.build(input_shape=(4, original_dim))
    model.load_weights(encoder_path)
elif type_encoder == "PAE":
    model = PAE([hidden_dim, latent_dim, original_dim], switch=True)
    model.build(input_shape=(4, original_dim))
    print(sys.argv)
    b_path = sys.argv[8]
    scaler_path = sys.argv[9]
    model.load_weights_model([encoder_path, decoder_path,
                              b_path, scaler_path])

x_train = np.load(path_data + "x_train.npy")
y_train = np.load(path_data + "y_train.npy")

#train, test = tf.keras.datasets.mnist.load_data()
#(x_train, y_train) = train
#(x_test, y_test) = test

x_train = model.preformat(x_train)
x_train = model.preprocessing(x_train)
print(x_train.shape)
z_test= np.array(model.encode(x_train))

# Start visualization
name_labels = ["Label 0", "Label 1", "Label 2", "Label 3", "Label 4",
               "Label 5", "Label 6", "Label 7", "Label 8", "Label 9"]
D = Distribution(z_test[:,0], z_test[:,1], y_train, name_labels)

# Load tree
#path_tree = "compostela_outputs/0/json/" + str(num1) + "-" + str(num2) + ".json"
#with open(path_tree) as f:
#    data = json.load(f)

AE = AutoEncoder_AAE(model, 2)
#AE.fit_reduce(x_test)
fig = Figure()
#fig.set_title("Model " + type_encoder)
fig.set_title("Autoencoder")
if path_tree is None:
    fig.plot_latent(AE, D)
else:
     with open(path_tree) as f:
         data = json.load(f)
     fig.plot_latent(AE, D, tree=data, draw_are=True, show_image=True)
