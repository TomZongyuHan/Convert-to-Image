# Import library and methods
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


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
def dimensionalityReduce(normalizedDataset, drName, returnMethod):
    print("Dimension Reducing......")
    # Transpose the dataset for dimensionality reduction
    # normalizedDatasetT = np.transpose(normalizedDataset)
    normalizedDatasetT = normalizedDataset.transpose()

    # Implement and run dimensionality reduction methods
    if drName == 'pca':
        drMethod = PCA(n_components=0.95)
        drDatasetT = drMethod.fit_transform(normalizedDatasetT)
        labels = normalizedDataset.columns.values
    elif drName == 'kpca':
        drDataset = 1
    elif drName == 'tsne':
        drDataset = 1
    elif drName == 'phate':
        drDataset = 1
    else:
        print("????? Please enter a correct normalize name ?????")

    # Transpose the dataset for results
    # drDataset = np.transpose(drDatasetT)
    drDataset = drDatasetT.transpose()

    # Return the result
    if returnMethod:
        return [drMethod, normalizedDataset]
    else:
        return [drDataset, labels]


# # Test
# from methods import dataCleanAndNormalize
#
# filepath = '../originalDatasets/' + 'yan-RowCount.csv'
# normalizedDataset = dataCleanAndNormalize.dataCleanAndNormalize(filepath, True, "linnorm")
#
# drDataset = dimensionalityReduce(normalizedDataset, 'pca')
# print(drDataset)
# print(normalizedDataset.shape)
# print(drDataset.shape)
