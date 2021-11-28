# Import library and methods
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
import phate
import umap


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
    # Transpose the dataset for dimensionality reduction
    normalizedDatasetT = normalizedDataset.transpose()
    labels = normalizedDataset.columns.tolist()
    featureNum = len(labels)
    sampleNum = len(normalizedDatasetT.iloc[1])
    componentNum = int((min(featureNum, sampleNum) - 1) * 0.5)
    for i in range(len(labels)):
        name = labels[i].split('.')
        if len(name) > 1:
            labels[i] = ''.join([str(elem) for elem in name[:-1]])
        else:
            labels[i] = name[0]

    # Implement and run dimensionality reduction methods
    if drName == 'pca':
        drMethod = PCA(n_components=componentNum)
        drDatasetT = drMethod.fit_transform(normalizedDatasetT.values)
    elif drName == 'kpca':
        drMethod = KernelPCA(n_components=componentNum, kernel='cosine', n_jobs=-1)
        drDatasetT = drMethod.fit_transform(normalizedDatasetT.values)
    elif drName == 'tsne':
        drMethod = TSNE(n_components=2, n_jobs=-1, method='exact')
        drDatasetT = drMethod.fit_transform(normalizedDatasetT.values)
    elif drName == 'phate':
        drMethod = phate.PHATE(n_components=2, n_jobs=-1)
        drDatasetT = drMethod.fit_transform(normalizedDatasetT.values)
    elif drName == 'umap':
        drMethod = umap.UMAP(n_components=componentNum)
        drDatasetT = drMethod.fit_transform(normalizedDatasetT.values)
    else:
        print("????? Please enter a correct normalize name ?????")

    # Transpose the dataset for results
    drDataset = drDatasetT.transpose()

    # Return the result
    return [drName, normalizedDataset, drDataset, labels]
