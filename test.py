# Test for 1 pipeline
# Import library and methods
from methods import dataCleanAndNormalize
from methods import dimensionalityReduce
from methods import imageConvert
from methods import imageEnhance
from methods import CNNTrain
from methods import calculateAccuracy


# test method include 1 pipeline
# Input:
#   filename: string value, the name of csv file (include .csv)
#       dataset file should be put in ./originalDatasets
#   isRowCount: boolean value,
#       if dataset is row count, use True
#       if dataset is not row count, use False
# Output:
#   result files will be store in ./results
def test(filename, isRowCount):
    # Set file path
    filepath = 'originalDatasets/' + filename

    # Set all methods name list
    normNames = ['linnorm']
    drNames = ['pca']
    icNames = ['deepinsight']
    CNNNames = ['vgg16']

    # Run all methods and output results
    finishNum = 0  # use a number to calculate how many method have finished
    allNum = len(normNames) * len(drNames) * len(icNames) * len(CNNNames)  # calculate how many methods exist
    for normName in normNames:
        # Data clean and normalize
        normalizedDataset = dataCleanAndNormalize.dataCleanAndNormalize(filepath, isRowCount, normName)
        for drName in drNames:
            # Dimensionality reduce
            drDataset = dimensionalityReduce.dimensionalityReduce(normalizedDataset, drName)
            for icName in icNames:
                # imageDataset = imageConvert(drDataset, icName) # Image convert
                # enhancedDataset = imageEnhance(imageDataset) # Image enhance
                for CNNName in CNNNames:
                    # result = CNNTrain(enhancedDataset, CNNName) # CNN train
                    # calculateAccuracy(result) # Calculate accuracy
                    finishNum += 1
                    print('----- ' +
                          normName + '-' +
                          drName + '-' +
                          icName + '-' +
                          CNNName + ' finish ' +
                          str(finishNum) + '/' + str(allNum))

    # Test return
    return drDataset


# Run test
filename = 'yan-RowCount.csv'
testRes = test(filename, True)
print("shape of drDataset: " + str(testRes.shape))
print(testRes)
