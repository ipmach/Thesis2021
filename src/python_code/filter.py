#!/usr/bin/env python3
from DataSets.getData import GetData
from tqdm import tqdm
import numpy as np
import sys

folder_path = sys.argv[1]
dataset_train = sys.argv[2]
option = sys.argv[3]

# Load data
x_data, y_data = GetData.get_ds(dataset_train)

# Options
if option == "Permutation":
    permutations = np.load(sys.argv[4])
    value = int(sys.argv[5])
    data_index_0 = np.array(list(map(lambda x: (x == value), y_data)))
    data_index_0 = data_index_0.astype(int)
    print(permutations)
    for j, i in enumerate(tqdm(permutations)):
        data_index_1 = np.array(list(map(lambda x: (x == i), y_data)))
        data_index_1 = data_index_1.astype(int)
        data_index = data_index_0 + data_index_1
        data_index = np.nonzero(data_index)[0]
        y_data_ = y_data[data_index]
        x_data_ = x_data[data_index]
        # Save each iteration
        x_train, y_train, x_test, y_test = GetData.split_data(x_data_,
                                                              y_data_)
        np.save(folder_path + "/" + str(j) + "/y_train.npy", y_train)
        np.save(folder_path + "/" + str(j) + "/x_train.npy", x_train)
        np.save(folder_path + "/" + str(j) + "/y_test.npy", y_test)
        np.save(folder_path + "/" + str(j) + "/x_test.npy", x_test)
elif option == "FilterValues":
    for j, value in enumerate(sys.argv[4:]):
        if j == 0:
            data_index = np.array(list(map(lambda x: (x == int(value)), y_data)))
            data_index = data_index.astype(int)
        else:
            data_index_1 = np.array(list(map(lambda x: (x == int(value)), y_data)))
            data_index_1 = data_index_1.astype(int)
            data_index = data_index + data_index_1
    data_index = np.nonzero(data_index)[0]
    y_data = y_data[data_index]
    x_data = x_data[data_index]

    x_train, y_train, x_test, y_test = GetData.split_data(x_data,
                                                          y_data)
    np.save(folder_path + "/y_train.npy", y_train)
    np.save(folder_path + "/x_train.npy", x_train)
    np.save(folder_path + "/y_test.npy", y_test)
    np.save(folder_path + "/x_test.npy", x_test)









