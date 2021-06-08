from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os


def get_data(path):
    """
    Get data from path
    :param path: experiment folder path
    :return: DataFrame
    """
    elements = []
    directories_m = next(os.walk(path))[1]
    directories_m.remove('gradient_tape')
    directories_m.remove('train')
    directories_m.remove('plugins')
    for model in directories_m:
        path20 = path + "/" + model
        directories_z = next(os.walk(path20))[1]
        element_m = [model]
        for latent in directories_z:
            element_z = element_m.copy()
            element_z.append(latent.split('_')[-1])
            path2 = path20 + "/" + latent
            directories_h = next(os.walk(path2))[1]
            for hidden in directories_h:
                element_h = element_z.copy()
                element_h.append(hidden.split('_')[-1])
                path3 = path2 + "/" + hidden
                datasets = next(os.walk(path3))[1]
                for dataset in datasets:
                    element_d = element_h.copy()
                    element_d.append(dataset)
                    path4 = path3 + "/" + dataset
                    epochs = next(os.walk(path4))[2]
                    for epoch in epochs:
                        if model == 'RandNet':
                            element_e = element_d.copy()
                            aux = epoch.replace("Epoch_", "").replace(".npy", "").split(" ")
                            element_e.append(aux[0])
                            #print(path4 + "/" + epoch)
                            score = np.load(path4 + "/" + epoch)
                            element_e.append(score)
                            element_e.append(aux[1])
                            elements.append(element_e)
                        else:
                            element_e = element_d.copy()
                            element_e.append(epoch.replace("Epoch_", "").replace(".npy", ""))
                            #print(path4)
                            score = np.load(path4 + "/" + epoch)
                            element_e.append(score)
                            element_e.append("None")
                            elements.append(element_e)
    return pd.DataFrame(elements, columns=["Model", "Latent Space", "Hidden Space",
                                           "DataSet","Epoch", "Values", "Submodel"])


def process_RandNet(df):
    """
    Process RandNet (join all Rand AE together)
    :param df: dataframe
    :return: dataframe
    """
    df_2 = []
    dfRa = df[df['Model'] == 'RandNetCNN']
    for l in dfRa['Latent Space'].unique():
        dfR = dfRa[dfRa['Latent Space'] == l]
        for h in dfR['Hidden Space'].unique():
            auxR = dfR[dfR['Hidden Space'] == h]
            for e in auxR['Epoch'].unique():
                auxR2 = auxR[auxR['Epoch'] == e]
                for d in auxR2['DataSet'].unique():
                    auxR3 = auxR2[auxR2['DataSet'] == d]
                    df_2.append(['RandNetCNN', l, h, d, e,
                                 np.median(np.array(list(dict(auxR3['Values']).values())), axis=0),
                                'None'])

    df_2 = pd.DataFrame(df_2, columns=['Model', 'Latent Space', 'Hidden Space',
                                       'DataSet', 'Epoch', 'Values', 'Submodel'])
    df = df[df['Model'] != 'RandNetCNN']
    df = pd.concat([df, df_2], ignore_index=True)
    return df[["Model", "Latent Space", "Hidden Space", "DataSet", "Epoch", "Values"]]


def separate_ood(df, path_json='src/python_code/settings.json'):
    """
    Add ood column in the dataframe
    :param df: dataframe
    :param path_json: setting file
    :return: dataframe
    """
    settings = json.load(open(path_json))["OOD"]["Gather_Data"]
    names_ood = settings["Set_DataSets"][int(settings["Choose_set"])]["OOD"]
    ood = []
    for name_ood in names_ood:
        ood = ood + [name_ood + ' BinaryCross', name_ood + ' BinaryCross Likehood', name_ood + ' BinaryCross Disc',
                     name_ood + ' BinaryCross LogNormLikelihood', name_ood + ' BinaryCross Mahalanobis']
    """
    ood = ['FashionMnist BinaryCross', 'MNIST-C BinaryCross',
                         'FashionMnist BinaryCross Likehood', 'MNIST-C BinaryCross Likehood',
                         'FashionMnist BinaryCross Likehood','MNIST-C BinaryCross Likehood',
                          'MNIST-C BinaryCross Disc', 'FashionMnist BinaryCross Disc']
    """
    df["ood"] = df['DataSet'].map(lambda x: 1 if x in ood else 0)
    return df


def remove_latent(df, path_json='src/python_code/settings.json'):
    """
    Remove latent data from df
    :param df: dataframe
    :param path_json: setting file
    :return: new dataframe
    """
    settings = json.load(open(path_json))["OOD"]["Gather_Data"]
    names_ood = settings["Set_DataSets"][int(settings["Choose_set"])]["OOD"]
    methods = settings["Feature_methods"]
    for method in methods:
        for name_ood in names_ood:
            df = df[df['DataSet'] != name_ood + ' BinaryCross ' + method]
        df = df[df['DataSet'] != 'Train OOD ' + method]
        df = df[df['DataSet'] != 'Test OOD ' + method]
    """
    df = df[df['DataSet'] != 'FashionMnist BinaryCross Likehood']
    df = df[df['DataSet'] != 'FashionMnist BinaryCross Disc']
    df = df[df['DataSet'] != 'MNIST-C BinaryCross Disc']
    df = df[df['DataSet'] != 'MNIST-C BinaryCross Likehood']
    df = df[df['DataSet'] != 'Train OOD Disc']
    df = df[df['DataSet'] != 'Test OOD Disc']
    df = df[df['DataSet'] != 'Train OOD Likehood']
    df = df[df['DataSet'] != 'Test OOD Likehood']
    """
    return df

def keep_feature(df, methods, path_json='src/python_code/settings.json'):
    """
    keep specific ood
    :param df: dataframe
    :param methods: methods name remove
    :param path_json: setting file
    :return: dataframe
    """
    df = df[df['DataSet'] != 'Train OOD BinaryCross']
    df = df[df['DataSet'] != 'Test OOD BinaryCross']
    settings = json.load(open(path_json))["OOD"]["Gather_Data"]
    names_ood = settings["Set_DataSets"][int(settings["Choose_set"])]["OOD"]
    for method in methods:
        df = df[df['DataSet'] != 'Train OOD ' + method]
        df = df[df['DataSet'] != 'Test OOD ' + method]
        for name_ood in names_ood:
            df = df[df['DataSet'] != name_ood + ' BinaryCross']
            df = df[df['DataSet'] != name_ood + ' BinaryCross ' + method]
    return df

"""
def keep_latent(df):
    df = df[df['DataSet'] != 'FashionMnist BinaryCross']
    df = df[df['DataSet'] != 'MNIST-C BinaryCross']
    df = df[df['DataSet'] != 'Train OOD BinaryCross']
    df = df[df['DataSet'] != 'Test OOD BinaryCross']
    df = df[df['DataSet'] != 'Train OOD Likehood']
    df = df[df['DataSet'] != 'Test OOD Likehood']
    df = df[df['DataSet'] != 'MNIST-C BinaryCross Likehood']
    df = df[df['DataSet'] != 'FashionMnist BinaryCross Likehood']
    return df


def keep_like(df):
    df = df[df['DataSet'] != 'FashionMnist BinaryCross']
    df = df[df['DataSet'] != 'MNIST-C BinaryCross']
    df = df[df['DataSet'] != 'Train OOD BinaryCross']
    df = df[df['DataSet'] != 'Test OOD BinaryCross']
    df = df[df['DataSet'] != 'Train OOD Disc']
    df = df[df['DataSet'] != 'Test OOD Disc']
    df = df[df['DataSet'] != 'MNIST-C BinaryCross Disc']
    df = df[df['DataSet'] != 'FashionMnist BinaryCross Disc']
    return df
"""

def auc(x, df):
    """
    AucRoc metric, add new column to the dataframe
    :param x: row of the dataframe
    :param df: Dataframe
    :return: new value in row
    """
    no_ood = df[df['ood'] == 0]
    no_ood = no_ood[no_ood['Epoch'] == x.Epoch]
    no_ood = no_ood[no_ood['Model'] == x.Model]
    no_ood = no_ood[no_ood['Latent Space'] == x['Latent Space']]
    no_ood = no_ood[no_ood['Hidden Space'] == x['Hidden Space']]
    aux = list(dict(no_ood.Values).values())
    try:
        #no_ood = np.append(aux[0], aux[1])
        """
        if aux[0] > 2:
            no_ood = np.mean(aux[0].reshape((-1, 32*32)), axis=1)
            data = np.concatenate([np.mean(x['Values'].reshape((-1, 32*32)), axis=1), no_ood])
        else:
            no_ood = np.mean(aux[0], axis=1)
            data = np.concatenate([np.mean(x['Values'], axis=1), no_ood])
        labels = np.concatenate([np.ones(x['Values'].shape[0]), np.zeros(no_ood.shape[0])])
        return roc_auc_score(labels, data)
        """
        if len(aux[0].shape) > 2:
            no_ood = np.mean(aux[0].reshape((-1, 32*32)), axis=1)
            data = np.concatenate([np.mean(x['Values'].reshape((-1, 32*32)), axis=1), no_ood])
            labels = np.concatenate([np.ones(x['Values'].shape[0]), np.zeros(aux[0].shape[0])])
        else:
            #print("hola")
            if len(x.Values.shape) > 1:
                no_ood = np.mean(aux[0], axis=1)
                data = np.concatenate([np.mean(x['Values'], axis=1), no_ood])
                labels = np.concatenate([np.ones(x['Values'].shape[0]), np.zeros(aux[0].shape[0])])
            else:
                data = np.concatenate([x['Values'], aux[0]])
                labels = np.concatenate([np.zeros(2000), np.ones(2000)])
        return roc_auc_score(labels, data)
    except Exception as e:
        print("Epoch {} Model {} Latent {} Hidden {}".format(x.Epoch, x.Model,
                                                             x['Latent Space'], x['Hidden Space']))
        print(e)
        return None


def gather_data(path, save_file=None, path_json='src/python_code/settings.json'):
    """
    Gather data from different experiments
    :param path: path of the experiments
    :param save_file: path if you want to save data as csv (Default None)
    :param path_json: setting file
    :return: dataframe
    """
    experiments = next(os.walk(path))[1]
    settings = json.load(open(path_json))["OOD"]["Gather_Data"]
    methods = settings["Feature_methods"]
    for j, experiment in enumerate(tqdm(experiments)):
        df = get_data(path + experiment + '/logs')
        df = process_RandNet(df)
        df = separate_ood(df, path_json=path_json)
        df2 = remove_latent(df)
        df2['auc'] = df2.apply(lambda x: auc(x, df2) if x.ood == 1 else None, axis=1)
        print("methods ", methods)
        for method in methods:
            methods2 = methods.copy()
            methods2.remove(method)
            print("methods ", method, methods2)
            df3 = keep_feature(df, methods2, path_json=path_json)
            df3['auc'] = df3.apply(lambda x: auc(x, df3) if x.ood == 1 else None, axis=1)
            df2 = pd.concat([df2, df3])
        df = df2
        """
        df3 = keep_latent(df)
        df3['auc'] = df3.apply(lambda x: auc(x, df3) if x.ood == 1 else None, axis=1)
        df4 = keep_like(df)
        df4['auc'] = df4.apply(lambda x: auc(x, df4) if x.ood == 1 else None, axis=1)
        df = pd.concat([df2, df3, df4])
        """
        #df['Metric use'] = df['Values'].apply(lambda x: np.mean(np.array(x).reshape(-1)))
        df['Epoch'] = df['Epoch'].apply(lambda x: int(x))
        df = df[['Model', 'Latent Space', 'Hidden Space', 'DataSet', 'Epoch', 'ood', 'auc']]

        if j == 0:
            final_df = df.copy()
        else:
            final_df = pd.concat([final_df, df])
    if save_file is not None:
        final_df.to_csv(save_file)

    return final_df
