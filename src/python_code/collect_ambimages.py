#!/usr/bin/env python3

from tqdm import tqdm
import pandas as pd
import json
import sys
import os
def parent_search(ambiguous, x):
    """
    Get parents ID
     :param ambiguous: dataframe data
     :param x: element in dataframe
     :return: ID parent or None if doesnt have parent
    """
    if x is not None:
        return list(ambiguous[ambiguous['path png'] == x]['ID'].items())[0][1]
    return None

def csv_generator(path, path_json, delta_1, delta_2, ID):
    """
    Generathe csv from one path
    :param path: path of the images
    :param path_json: path of the json file
    :return: csv of a specific folder
    """
    directories = next(os.walk(path))[1]
    dataset = []
    path_ = path.replace("imgs/", "")  # temporal solution
    for dir_ in tqdm(directories):
        directories_img = next(os.walk(path + "/" + dir_))[2]
        labels = dir_.split("-")
        aux_path = "/" + labels[0] + "-" + labels[1] + ".json"
        json_file = json.load(open(path_json + aux_path))
        for dir_img in directories_img:
            if dir_img == "'tree" or dir_img == "tree.pdf":
                continue
            img_nr = dir_img.split('.')[0]
            parent = json_file[img_nr]['father']
            disc_1 = json_file[img_nr]['discriminator label ' + str(labels[0])]
            disc_2 = json_file[img_nr]['discriminator label ' + str(labels[1])]
            parent = path_ + 'imgs/' + dir_ + '/' + str(parent) + '.png' if parent is not None else None
            dataset.append(['A' + str(ID), labels[0], labels[1], path_ + 'imgs/' + dir_ + '/' + dir_img,
                            path_ + 'numpy/data' + str(labels[0]) + '_' + str(labels[1]) + '.npy', img_nr,
                            delta_1, delta_2, disc_1, disc_2, path_ + 'json' + aux_path, parent, 'Ambiguous'])
            ID += 1

    # Convert in dataframe
    ambiguous = pd.DataFrame(dataset, columns=['ID', 'label 0', 'label 1', 'path png', 'path numpy', 'index numpy',
                                               'delta 1', 'delta 2', 'discriminator value 1', 'discriminator value 2',
                                               'json_file', 'parent', 'type'])

    # Get ID parents
    ambiguous['parent2'] = ambiguous['parent'].apply(lambda x: parent_search(x))
    return ambiguous, ID
# Parameters
ambiguous_path = sys.argv[1]
csv_path = sys.argv[2]
delta_1 = sys.argv[3]
delta_2 = sys.argv[4]
# Do the csv
ID = 0  # needed to create the primary key
main_directories = next(os.walk(ambiguous_path))[1]
initalized = False  # dataframe is initialized or not
for directory in tqdm(main_directories, position=0):
    if 'outputs' in next(os.walk(ambiguous_path + '/' + directory))[1]:
        output_directory = ambiguous_path + '/' + directory + '/outputs'
        sub_directories = next(os.walk(output_directory))[1]
        for sub_directory in sub_directories:
            if not initalized:
                path_ = output_directory + '/' + sub_directory
                df, ID = csv_generator(path_ + '/imgs', path_ + '/json', delta_1, delta_2, ID)
            else:
                df_aux, ID = csv_generator(path_ + '/imgs', path_ + '/json', delta_1, delta_2, ID)
                df = pd.concat([df, df_aux.copy(True)])
# Save csv
df.to_csv(csv_path, index=False)
