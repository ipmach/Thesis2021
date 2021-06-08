#!/usr/bin/env python3
from DataSets.getData import GetData
import numpy as np
import sys

folder_path = sys.argv[1]
dataset_train = sys.argv[2]

x_data, y_data = GetData.get_ds(dataset_train)
x_train, y_train, x_test, y_test = GetData.split_data(x_data,
                                                      y_data)

np.save(folder_path + "/y_train.npy", y_train)
np.save(folder_path + "/x_train.npy", x_train)
np.save(folder_path + "/y_test.npy", y_test)
np.save(folder_path + "/x_test.npy", x_test)

if len(sys.argv) > 3:
    list_names = ""
    for j, dataset_ood in enumerate(sys.argv[3:]):
        x_data, _ = GetData.get_ds(dataset_ood)
        np.save(folder_path + "/odd_" + str(j) + ".npy", x_data)
        list_names = list_names + " & " + dataset_ood
    print("Saving names in :", folder_path + "/names.txt")
    with open(folder_path + "/names.txt", "w") as text_file:
        text_file.write(list_names)




