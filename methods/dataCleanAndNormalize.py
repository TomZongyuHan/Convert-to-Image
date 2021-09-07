# Import library and methods


# Call R file to clean and normalize dataset
# Input:
#   filepath: string value, the path of csv file
#   isRowCount: boolean value, 
#       if dataset is row count, use True
#       if dataset is not row count, use False
#   normName: string value, the name of normalization method
# Output:
#   normalizedDataset: the variable store normalized dataset
def dataCleanAndNormalize(filepath, isRowCount, normName):
    # Import R code method
    processDataset = 1

    # Store processed dataset
    normalizedDataset = processDataset(filepath, isRowCount, normName)

    # Return processed dataset
    return normalizedDataset
