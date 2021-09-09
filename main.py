# Import library and methods
import sys

from methods import dataCleanAndNormalize
from methods import dimensionalityReduce
from methods import imageConvert
from methods import imageAugumentation
from methods import CNNTrain
from methods import calculateAccuracy

# Main method include all pipeline
# Input:
#   filename: string value, the name of csv file (include .csv)
#       dataset file should be put in ./originalDatasets
#   isRowCount: boolean value, 
#       if dataset is row count, use True
#       if dataset is not row count, use False
# Output:
#   result files will be store in ./results
def main(filename, isRowCount):
    # Set file path
    filepath = './originalDatasets/' + filename
    
    # Set all methods name list
    normNames = ['linnorm', 'scran', 'tmm', 'scone', 'cpm', 'seurat']
    drNames = ['pca', 'kpca', 'tsne', 'phate']
    icNames = ['deepinsight', 'cpcr', 'gaf']
    CNNNames = ['alexnet', 'vgg16', 'squeezenet', 'resnet', 'densenet']

    # Run all methods and output results
    finishNum = 0 # use a number to calculate how many method have finished
    allNum = len(normNames) + len(drNames) + len(icNames) + len(CNNNames) # calculate how many methods exist
    for normName in normNames:
        # Data clean and normalize
        normalizedDataset = dataCleanAndNormalize(filepath, isRowCount, normName)
        for drName in drNames:
            # Dimensionality reduce
            drDataset = dimensionalityReduce(normalizedDataset, drName)
            for icName in icNames:
                imageDataset = imageConvert(drDataset, icName) # Image convert
                enhancedDataset = imageAugumentation(imageDataset) # Image enhance
                for CNNName in CNNNames:
                    result = CNNTrain(enhancedDataset, CNNName) # CNN train
                    calculateAccuracy(result) # Calculate accuracy
                    finishNum += 1
                    print('----- ' + 
                        normName + '-' + 
                        drName + '-' + 
                        icName + '-' + 
                        CNNName + ' finish ' + 
                        finishNum + '/' + allNum)

# main function entry
if __name__ == '__main__':
    main(sys.argv[0], sys.argv[1])
