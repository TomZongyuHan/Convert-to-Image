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
import numpy as np


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
    normNames = ['linnorm', 'scran', 'tmm', 'scone', 'cpm', 'seurat']
    drNames = ['pca', 'kpca', 'tsne', 'phate']
    icNames = ['deepinsight', 'cpcr', 'gaf']
    CNNNames = ['alexnet', 'vgg16', 'squeezenet', 'resnet', 'densenet']
    accNames = ['acc', 'f1_macro', 'f1_micro', 'f1_weighted', 'precision', 'recall']
    allNum = len(normNames) * len(drNames) * len(icNames) * len(CNNNames) * len(accNames)  # calculate how many methods exist

    # Handle the results file, skip completed method
    try:
        test_results = np.array(np.load('results/accuracies/testResults.npy', allow_pickle = True))
    except IOError:
        test_results = np.zeros((allNum, 1), dtype = np.float64) # allNum should be 2160

    # Run all methods and output results
    finishNum = 0  # use a number to calculate how many method have finished
    for normName in normNames:
        # The finishNum + 359 is last method line of this normalization
        # Check if need to skip one normalization method
        rst = checkSkipMethod('norm', finishNum + 359, test_results, [filepath, isRowCount, normName])
        if isinstance(rst, int):
            finishNum = rst + 1
            continue
        else:
            normalizedDataset = rst
        for drName in drNames:
            # Check if need to skip one dimensionality reduce method
            rst = checkSkipMethod('dr', finishNum + 89, test_results, [normalizedDataset, drName])
            if isinstance(rst, int):
                finishNum = rst + 1
                continue
            else:
                drResult = rst
            for icName in icNames:
                # Check if need to skip one image convert method
                rst = checkSkipMethod('ic', finishNum + 29, test_results, [drResult, icName])
                if isinstance(rst, int):
                    finishNum = rst + 1
                    continue
                else:
                    augmentedDataset = rst
                for CNNName in CNNNames:
                    # Check if need to skip one cnn method
                    rst = checkSkipMethod('cnn', finishNum + 5, test_results, [augmentedDataset, CNNName])
                    if isinstance(rst, int):
                        finishNum = rst + 1
                        continue
                    else:
                        results = rst
                    for accName in accNames:
                        methodName = normName + '-' + drName + '-' + icName + '-' + CNNName + '-' + accName
                        accuracy = calculateAccuracy(results, accName)  # Calculate accuracy
                        # Save the result in results/accuracies
                        test_results[finishNum] = accuracy
                        np.save('results/accuracies/testResults.npy', test_results)

                        # Print method name and finish num at terminal
                        finishNum += 1
                        print('----- ' +
                            methodName + ' finish ' +
                            str(finishNum) + '/' + str(allNum))   


# Check if skip this method and call the method
def checkSkipMethod(methodName, finishNum, test_results, params):
    skipFlag = float(test_results[finishNum]) != float(0)
    if methodName == 'norm':
        if not skipFlag:
            # If return the result of method, do not need to skip
            result = dataCleanAndNormalize(params[0], params[1], params[2]) # Data clean and normalize
        else:
            # If return the number of finishNum, need to skip these methods
            result = finishNum
    elif methodName == 'dr':
        if not skipFlag:
            result = dimensionalityReduce(params[0], params[1]) # Dimensionality reduce
        else:
            result = finishNum
    elif methodName == 'ic':
        if not skipFlag:
            imageDataset = imageConvert(params[0], params[1])  # Image convert
            result = imageAugumentation(imageDataset)  # Image Augmentation
        else:
            result = finishNum
    elif methodName == 'cnn':
        if not skipFlag:
            result = CNNTrain(params[0], params[1])  # CNN train
        else:
            result = finishNum
    return result


# Run test
filename = 'test-RowCount.csv'
test(filename, True)
