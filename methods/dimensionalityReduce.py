# Import library and methods
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
import phate


# Implement and run dimensionality reduction methods
# Input:
#   normalizedDataset: the variable store normalized dataset
#   drName: the name of dimensionality reduction method
#   returnMethod: boolean value,
#       if need to return the dr method, use True
#       if nedd to return processed dataset, use False
# Output:
#   drDataset: the variable of datasets after dimensionality reduce
#   list of drMethod and normalizedDataset: if use returnMethod
def dimensionalityReduce(normalizedDataset, drName):
    # print("Dimension Reducing......")
    # Transpose the dataset for dimensionality reduction
    # normalizedDatasetT = np.transpose(normalizedDataset)
    normalizedDatasetT = normalizedDataset.transpose()
    labels = normalizedDatasetT.columns.tolist()
    for i in range(len(labels)):
        name = labels[i].split('.')
        if len(name)>1:
            labels[i] = ''.join([str(elem) for elem in name[:-1]])
        else:
            labels[i] = name[0]

    # Implement and run dimensionality reduction methods
    if drName == 'pca':
        drMethod = PCA(n_components=2)
        drDatasetT = drMethod.fit_transform(normalizedDatasetT)
    elif drName == 'kpca':
        drMethod = KernelPCA(n_components=2, kernel='sigmoid')
        drDatasetT = drMethod.fit_transform(normalizedDatasetT)
    elif drName == 'tsne':
        drMethod = TSNE(n_components=2, n_jobs=-1)
        drDatasetT = drMethod.fit_transform(normalizedDatasetT)
    elif drName == 'phate':
        drMethod = phate.PHATE()
        drDatasetT = drMethod.fit_transform(normalizedDatasetT)
    else:
        print("????? Please enter a correct normalize name ?????")

    # Transpose the dataset for results
    # drDataset = np.transpose(drDatasetT)
    drDataset = drDatasetT.transpose()

    # Return the result
    return [drMethod, normalizedDataset, drDataset, labels]

# Test
# from methods import dataCleanAndNormalize
#
# filepath = '../originalDatasets/' + 'test-RowCount.csv'
# normalizedDataset = dataCleanAndNormalize.dataCleanAndNormalize(filepath, True, "linnorm")
#
# pca = dimensionalityReduce(normalizedDataset, 'pca', False)
# kpca = dimensionalityReduce(normalizedDataset, 'kpca', False)
# tsne = dimensionalityReduce(normalizedDataset, 'tsne', False)
# data_phate = dimensionalityReduce(normalizedDataset, 'phate', False)
# print(pca[0].shape)
# print(kpca[0].shape)
# print(tsne[0].shape)
# print(data_phate[0].shape)
