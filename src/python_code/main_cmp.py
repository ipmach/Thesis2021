#!/usr/bin/env python3
import sys

from compostela.algorithm import Compostela
from DataSets.getData import GetData
from Models.model_AAE import AAE
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os
from tqdm import tqdm

original_dim = 28 * 28
latent_dim = int(sys.argv[3])
hidden_dim = int(sys.argv[4])
num1 = 3
num2 = 6
nums = []
models_folder = sys.argv[1] + "/"
output_folder = sys.argv[2] + "/"

# Allow dynamic gpu memory growth
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


for num2 in range(10):
    for num1 in range(num2):
        nums.append([num1, num2])

#nums = [[0,4], [0,5], [0,6], [0,7], [0,8], [0,9], [1,4], [1,5], [1,6], [1,7], [1,8], [1,9]]

for j in range(5):
    local_output = output_folder + str(j) + "/"
    os.system("mkdir " + local_output)
    os.system("mkdir " + local_output + "tree_performance")
    os.system("mkdir " + local_output + "imgs")
    os.system("mkdir " + local_output + "json")
    os.system("mkdir " + local_output + "numpy")
    for i in tqdm(nums):
        [num1, num2] = i
        os.system("mkdir " + local_output + "imgs/" + str(num1) + "-" + str(num2))
        #(x_test, y_test), (_, _) = mnist.load_data()
        x_data, y_data = GetData.get_ds("MNIST")
        x_test, y_test, _, _ = GetData.split_data(x_data,  y_data)

        model = AAE([hidden_dim, latent_dim, original_dim], 10)
        path = models_folder
        model.encode_.load_weights(path + "model_AAE_" + str(num1) + "_" \
                           + str(num2) + "_encoder" + '$z_' + str(latent_dim) + "$h_" + str(hidden_dim) + ".h5")
        model.decode_.load_weights(path + "model_AAE_" + str(num1) + "_" \
                           + str(num2) + "_decoder" + '$z_' + str(latent_dim) + "$h_" + str(hidden_dim) + ".h5")
        model.discriminator_.load_weights(path + "model_AAE_" + str(num1) + "_" \
                           + str(num2) + "_discriminator" + '$z_' + str(latent_dim) + "$h_" + str(hidden_dim) + ".h5")
        """
        model.load_weights(path + "model_AAE_" + str(num1) + "_" \
                           + str(num2) + "_encoder" + '$z_' + str(latent_dim) \
                           + "$h_" + str(hidden_dim) + ".h5",
                           path + "model_AAE_" + str(num1) + "_" \
                           + str(num2) + "_decoder" + '$z_' + str(latent_dim) \
                           + "$h_" + str(hidden_dim) + ".h5",
                           path + "model_AAE_" + str(num1) + "_" \
                           + str(num2) + "_discriminator" + '$z_' + str(latent_dim) \
                           + "$h_" + str(hidden_dim) + ".h5")
        """

        test_index_0 = np.array(list(map(lambda x: (x==num1) , y_test)))
        test_index_1 = np.array(list(map(lambda x: (x==num2) , y_test)))
        test_index = test_index_0 + test_index_1
        test_index = np.nonzero(test_index)[0]
        y_test = y_test[test_index]
        x_test = x_test[test_index]

        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        z_test= model.encode(x_test)

        cmp = Compostela()

        cmp(model, x_test, y_test, num1, num2, local_output)
