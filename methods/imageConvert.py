# Import library and methods


# Transfer non-image data to image dataset
# Input:
#   drDataset: the variable of datasets after dimensionality reduce
#   icName: the name of image convert method
# Output:
#   imageDataset: the variable of image datasets
def imageConvert(drDatasets, icName):
    # Implement and run image convert methods
    if icName == 'deepinsight':
        imageDataset = 1
    elif icName == 'cpcr':
        imageDataset = 1
    elif icName == 'gaf':
        imageDataset = 1
    else:
        print("Please enter a correct image convert method name.")

    return imageDataset
