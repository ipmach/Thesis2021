#!/usr/bin/env python3
from Models.model_AAE_regularizer_train import AAE_train_re
from Models.model_Vanilla_CNN_train import VanillaCNN_train
from DataBase_Manager.db_manager import DB_manager_train
from DataBase_Manager.file_manager import FileManager
from Models.model_Vanilla_train import Vanilla_train
from Models.model_SAE_CNN_train import SAECNN_train
from Models.model_DAE_CNN_train import DAECNN_train
from Models.model_VAE_CNN_train import VAECNN_train
from Models.model_AAE_CNN_train import AAECNN_train
from Models.model_PAE_CNN_train import PAECNN_train
from Models.model_EAE_train import RandNet_train
from Models.model_SAE_train import SAE_train
from Models.model_DAE_train import DAE_train
from Models.model_VAE_train import VAE_train
from Models.model_AAE_train import AAE_train
from Models.model_PAE_train import PAE_train
from OOD.train_test import OODTest
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json
import copy
import sys

# CHECK GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

# CHECK CORRECT VERSION GPU
assert tf.__version__.startswith('2.'), "Tensorflow Version Below 2.0"

# GENERAL PARAMETERS
# Obtain database path
print(sys.argv)
sub_path = sys.argv[1]
folder_path = sys.argv[2]
path = sys.argv[2] + "/" + sub_path + "/"
model_save = sys.argv[3] + "/"
plot_save = sys.argv[4] + "/"
logs_save = sys.argv[5] + "/"
manager_folder = sys.argv[6]
# Hyperparameters
image_size = int(sys.argv[7])
hidden_dim = int(sys.argv[8])
latent_dim = int(sys.argv[9])
num_epochs = int(sys.argv[10])
batch_size = int(sys.argv[11])
learning_rate = float(sys.argv[12])
name = sys.argv[13]
name = name + "_" + str(latent_dim) + "_" + str(hidden_dim)
model_type = sys.argv[14]

# Load data
x_train = np.load(path + "x_train.npy")
y_train = np.load(path + "y_train.npy")
x_test = np.load(path + "x_test.npy")
y_test = np.load(path + "y_test.npy")

# Accessing settings
settings = json.load(open('src/python_code/settings.json'))["Train_script"]

# OOD
ood = bool(int(settings["do_ood"]))
ood_data = []
#oodTest = OODTest(["MNIST-C", "FashionMNIST"], "BinaryCross")
#oodTest = OODTest(["FashionMNIST"], "BinaryCross")
if ood:
    with open(folder_path + "/all/names.txt", "r") as text_file:
        names = text_file.readlines()
    names = names[0].split(' & ')[1:]
    oodTest = OODTest([name for name in names])
    for j, _ in enumerate(names):
        ood_data.append(np.load(path + "odd_" + str(j) + ".npy"))

# Preprocess data
x_train = x_train.astype(np.float32)

# Model Name
model_name = 'model_' + name

# FILE MANAGER
manager = FileManager(manager_folder)
description = "Training a {} with: \n image size {} \n hidden dim{} \n ".format(
model_type, image_size, hidden_dim) + "latent dim {} \n  num epochs {} \n batch size {}\n".format(
latent_dim, num_epochs, batch_size) + "learning rate {}".format(learning_rate)
m = manager.create_new_entrace(model_type + " trainng", description)
manager.insert_all_entrance(m, ["Logs", "Model", "Plot"],
                            [logs_save, model_save + model_name,
                             plot_save + model_name])
manager.save_entrance(m)
db = DB_manager_train(logs_save, model_name)
if ood:
    if model_type == "AAECNN":
        names_l = [name + " BinaryCross" for name in names]
        names_ld = [name + " Disc" for name in names_l]
        names_ll = [name + " Likehood" for name in names_l]
        names_llog = [name + " LogNormLikelihood" for name in names_l]
        names_maha = [name + " Mahalanobis" for name in names_l]
        names_ls = [name + " SVM1" for name in names_l]
        print(names_l + ["Test OOD BinaryCross",
                           "Train OOD BinaryCross"] + names_ld + ["Test OOD Disc", "Train OOD Disc"] +
                          names_ll + ["Test OOD Likehood", "Train OOD Likehood"] + names_ls)
        db.initialize_ood(logs_save, model_type, latent_dim, hidden_dim,
                          names_l + ["Test OOD BinaryCross",
                           "Train OOD BinaryCross"] + names_ld + ["Test OOD Disc", "Train OOD Disc"] +
                          names_ll + ["Test OOD Likehood", "Train OOD Likehood"] + names_ls +
                          names_llog + ["Test OOD LogNormLikelihood", "Train OOD LogNormLikelihood"] +
                          names_maha + ["Test OOD Mahalanobis", "Train OOD Mahalanobis"])
    elif model_type == "VAECNN" or model_type == "PAECNN":
        names_l = [name + " BinaryCross" for name in names]
        names_ll = [name + " Likehood" for name in names_l]
        names_ls = [name + " SVM1" for name in names_l]
        names_llog = [name + " LogNormLikelihood" for name in names_l]
        names_maha = [name + " Mahalanobis" for name in names_l]
        db.initialize_ood(logs_save, model_type, latent_dim, hidden_dim,
                          names_l + ["Test OOD BinaryCross", "Train OOD BinaryCross"] +
                          names_ll + ["Test OOD Likehood", "Train OOD Likehood"] + names_ls +
                          names_llog + ["Test OOD LogNormLikelihood", "Train OOD LogNormLikelihood"] +
                          names_maha + ["Test OOD Mahalanobis", "Train OOD Mahalanobis"])
    else:
        names_l = [name + " BinaryCross" for name in names] + ["Test OOD BinaryCross", "Train OOD BinaryCross"]
        db.initialize_ood(logs_save, model_type, latent_dim, hidden_dim, names_l)

# LOAD AND TRAIN MODEL
special_train = False  # If a model use a special train or not
#num_epochs = 186
if bool(int(settings["do_epochs_fix"])):
    num_epochs = int(settings["fix_epochs"])

if model_type == "Vanilla":  # Vanilla Autoencoder
    model = Vanilla_train([hidden_dim, latent_dim, image_size],
                          num_epochs, batch_size, learning_rate, db)
elif model_type == "SAE":  # Spartial Autoencoder
    model = SAE_train([hidden_dim, latent_dim, image_size],
                      num_epochs, batch_size, learning_rate, db)
elif model_type == "DAE":  # Denoise Autoencoder
    model = DAE_train([hidden_dim, latent_dim, image_size],
                      num_epochs, batch_size, learning_rate, db)
elif model_type == "VAE":  # Variational Autoencoder
    model = VAE_train([hidden_dim, latent_dim, image_size],
                      num_epochs, batch_size, learning_rate, db)
elif model_type == "AAE":  # Adversal Autoencoder
    train_buf = int(sys.argv[15])
    dropout = float(sys.argv[16])
    num_labels = 10  # MNIST constant (usefull with regularizer)
    model = AAE_train([hidden_dim, latent_dim, image_size],
                      num_epochs, batch_size, learning_rate, db,
                      num_labels, train_buf=train_buf, dropout=dropout)
elif model_type == "AAE_regularize":
    special_train = True  # Use his own method
    train_buf = int(sys.argv[15])
    dropout = float(sys.argv[16])
    num_labels = 10  # MNIST constant (usefull with regularizer)
    # Prepare for only two values
    values = np.unique(y_train)
    if len(values) > 2:
        raise Exception("Only two labels are allow in this format")
    model_name = 'model_AAE_' + str(values[0]) + "_" + str(values[1])
    model = AAE_train_re([hidden_dim, latent_dim, image_size],
                         num_epochs, batch_size, learning_rate, db,
                         num_labels, train_buf=train_buf, dropout=dropout)
    model.train(x_train, y_train, model_save, model_name, values[0], values[1])
elif model_type == "PAE":  # Probabilistic AutoEncoder
    model = PAE_train([hidden_dim, latent_dim, image_size],
                      num_epochs, batch_size, learning_rate, db)
elif model_type == "RandNet":  # Esemblence Autoencoder
    p_m = float(sys.argv[15])
    num = int(sys.argv[16])
    model = RandNet_train([hidden_dim, latent_dim, image_size],
                           p_m, num,
                           num_epochs, batch_size, learning_rate, db)
elif model_type == "VAEGAN":
    raise Exception("Model not implement it")
elif model_type == "VanillaCNN":
    filters = [64, 64, 32]  # GET FILTERS from json
    model = VanillaCNN_train(filters, hidden_dim, num_epochs,
                             batch_size, learning_rate, db)
elif model_type == "SAECNN":
    filters = [64, 64, 32]  # GET FILTERS from json
    model = SAECNN_train(filters, hidden_dim, num_epochs,
                         batch_size, learning_rate, db)
elif model_type == "DAECNN":
    filters = [64, 64, 32]  # GET FILTERS from json
    model = DAECNN_train(filters, hidden_dim, num_epochs,
                         batch_size, learning_rate, db)
elif model_type == "VAECNN":
    filters = [64, 64, 32]  # GET FILTERS from json
    model = VAECNN_train(filters, hidden_dim, num_epochs,
                         batch_size, learning_rate, db)
elif model_type == "AAECNN":
    train_buf = int(sys.argv[15])
    dropout = float(sys.argv[16])
    num_labels = 10
    filters = [64, 64, 32]  # GET FILTERS from json
    model = AAECNN_train(filters, [hidden_dim, latent_dim],
                         num_epochs, batch_size, learning_rate, db,
                         num_labels, train_buf=train_buf, dropout=dropout)
elif model_type == "PAECNN":
    filters = [64, 64, 32]  # GET FILTERS from json
    model = PAECNN_train(filters, hidden_dim, num_epochs,
                         batch_size, learning_rate, db)

elif model_type == "RandNetCNN":  # Esemblence Autoencoder
    p_m = float(sys.argv[15])
    num = int(sys.argv[16])
    filters = [64, 64, 32]  # GET FILTERS from json
    model = RandNet_train([hidden_dim, latent_dim, image_size],
                           p_m, num, num_epochs, batch_size,
                           learning_rate, db, cnn=True, filters=filters)
else:
    print(sys.argv)
    print(model_type)
    raise Exception("Model not found")

if not special_train:  # Normal format of training
    model.train(copy.copy(x_train), model_save, model_name,
                ood_data=ood_data, x_test=x_test, ood_op=oodTest)
    #db.compress(model_name)

# SAVE DISTRIBUTION
if model_type != "RandNet" and model_type != 'RandNetCNN':
    x_test = model.preprocessing(x_test)
    x_test = model.preformat(x_test)
    z = model.encode(x_test)
    plt.figure(figsize=(6, 6))
    try:
        plt.scatter(z[:, 0], z[:, 1], c=y_test)
    except Exception:
        plt.scatter(z, np.zeros(len(z)), c=y_test)
    plt.colorbar()
    plt.savefig(plot_save + model_name)






