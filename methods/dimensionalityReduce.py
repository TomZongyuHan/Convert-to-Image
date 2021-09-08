# Import library and methods
import numpy as np
from sklearn.decomposition import PCA


# Implement and run dimensionality reduction methods
# Input:
#   normalizedDataset: the variable store normalized dataset
#   drName: the name of dimensionality reduction method
# Output:
#   drDataset: the variable of datasets after dimensionality reduce
def dimensionalityReduce(normalizedDataset, drName):
    # Transpose the dataset for dimensionality reduction
    normalizedDatasetT = np.transpose(normalizedDataset)

    # Implement and run dimensionality reduction methods
    if drName == 'pca':
        pca = PCA(n_components=0.95)
        drDatasetT = pca.fit_transform(normalizedDatasetT)
    elif drName == 'kpca':
        drDataset = 1
    elif drName == 'tsne':
        drDataset = 1
    elif drName == 'phate':
        drDataset = 1
    else:
        print("Please enter a correct normalize name.")

    # Transpose the dataset for results
    drDataset = np.transpose(drDatasetT)

    return drDataset


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
