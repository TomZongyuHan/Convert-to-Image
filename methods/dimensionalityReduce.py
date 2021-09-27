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
    labels = normalizedDataset.columns.tolist()
    componentNum = min(len(labels), len(normalizedDatasetT.iloc[1])) - 1
    
    for i in range(len(labels)):
        name = labels[i].split('.')
        if len(name) > 1:
            labels[i] = ''.join([str(elem) for elem in name[:-1]])
        else:
            labels[i] = name[0]

    # Implement and run dimensionality reduction methods
    if drName == 'pca':
        drMethod = PCA(n_components=None)
        drDatasetT = drMethod.fit_transform(normalizedDatasetT.values)
    elif drName == 'kpca':
        drMethod = KernelPCA(n_components=None, kernel='cosine', n_jobs=-1)
        drDatasetT = drMethod.fit_transform(normalizedDatasetT.values)
    elif drName == 'tsne':
        drMethod = TSNE(n_components=componentNum, n_jobs=-1, method='exact')
        drDatasetT = drMethod.fit_transform(normalizedDatasetT.values)
    elif drName == 'phate':
        drMethod = phate.PHATE(n_components=componentNum, n_jobs=-1)
        drDatasetT = drMethod.fit_transform(normalizedDatasetT.values)
    else:
        print("????? Please enter a correct normalize name ?????")

    # Transpose the dataset for results
    # drDataset = np.transpose(drDatasetT)
    drDataset = drDatasetT.transpose()

    # Return the result
    return [drName, normalizedDataset, drDataset, labels]


# Test
# from methods import dataCleanAndNormalize
#
# filepath = '../originalDatasets/' + 'TabulaMuris_Thymus_10X-RowCount.csv'
# normalizedDataset = dataCleanAndNormalize.dataCleanAndNormalize(filepath, True, "linnorm")
#
# res = dimensionalityReduce(normalizedDataset, 'pca')
# print(res[0])
# print(res[1])
# print(res[2])
# print(res[3])
