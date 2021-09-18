# Test for the pipeline
# Import library and methods
import sys
sys.path.append('./methods/')
from dataCleanAndNormalize import dataCleanAndNormalize
from dimensionalityReduce import dimensionalityReduce
from imageConvert import imageConvert
from imageAugumentation import imageAugumentation
from CNNTrain import CNNTrain
from calculateAccuracy import calculateAccuracy


# test methods included in the pipeline
# Input:
#   filename: string value, the name of csv file (include .csv)
#       dataset file should be put in ./originalDatasets
#   isRowCount: boolean value,
#       if dataset is row count, use True
#       if dataset is not row count, use False
# Output:
#   result files will be showed in commond line
def test(filename, isRowCount):
    # Set file path
    filepath = 'originalDatasets/' + filename

    # Set all methods need to be test
    normNames = ['phate']
    drNames = ['pca']
    icNames = ['deepinsight']
    CNNNames = ['vgg16']
    accNames = ['acc']

    # Run all methods and output results
    finishNum = 0  # use a number to calculate how many method have finished
    allNum = len(normNames) * len(drNames) * len(icNames) * len(CNNNames) * len(accNames)  # calculate how many methods exist
    for normName in normNames:
        # Data clean and normalize
        normalizedDataset = dataCleanAndNormalize(filepath, isRowCount, normName)
        for drName in drNames:
            # Dimensionality reduce
            for icName in icNames:
                drResult = dimensionalityReduce(normalizedDataset, drName, icName)  # Dimensionality reduce
                imageDataset = imageConvert(drResult, icName)  # Image convert
                augmentedDataset = imageAugumentation(imageDataset)  # Image Augmentation
                for CNNName in CNNNames:
                    results = CNNTrain(augmentedDataset, CNNName)  # CNN train
                    for accName in accNames:
                        calculateAccuracy(results, accName)  # Calculate accuracy
                        finishNum += 1
                        print('----- ' +
                              normName + '-' +
                              drName + '-' +
                              icName + '-' +
                              CNNName + ' finish ' +
                              str(finishNum) + '/' + str(allNum))


# Run test
filename = 'test-RowCount.csv'
test(filename, True)
