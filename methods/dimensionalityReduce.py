# Import library and methods


# Implement and run dimensionality reduction methods
# Input:
#   normalizedDataset: the variable store normalized dataset
#   drName: the name of dimensionality reduction method
# Output:
#   drDataset: the variable of datasets after dimensionality reduce
def dimensionalityReduce(normalizedDataset, drName):
    # Implement and run dimensionality reduction methods
    if drName == 'pca':
        drDataset = 1
    elif drName == 'kpca':
        drDataset = 1
    elif drName == 'tsne':
        drDataset = 1
    elif drName == 'phate':
        drDataset = 1
    else:
        print("Please enter a correct normalize name.")

    return drDataset
