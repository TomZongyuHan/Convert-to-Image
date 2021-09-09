# Import library and methods
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import numpy as np
import pandas as pd


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
    # Transform variable 'isRowCount' to String
    if (isRowCount):
        isRowCount = 'TRUE'
    else:
        isRowCount = 'FALSE'

    # Prepare R variable
    filepathpy = filepath
    filepath = '\"' + filepath + '\"'
    isRowCount = '\"' + isRowCount + '\"'
    normName = '\"' + normName + '\"'

    # Transform python String to R String
    robjects.r('''
        path  <- gsub("to", "",''' + filepath + ''')
        isRowCount <- gsub("to", "",''' + isRowCount + ''')
        normname <- gsub("to", "",''' + normName + ''')
    ''')

    # Import R's packages
    importr('Linnorm')  # for linnorm
    importr('scone')  # for scone/TTM/scran
    importr('edgeR')  # for cpm
    importr('Seurat')  # for seurat

    robjects.r("""
            # This function used to read, clean and normalize data
            # Input:
            #   path: data csv file path
            #   normName: normalization method name, should be linnorm/scone/ttm/scran/cpm/seurat
            # Rerutn:
            #   processed data: rows - genes, columns - cells
            processDataset <- function(path, isRowCount, normname) {
            # Read the CSV file and convert it to matrix format
            rowdata <- read.table(path, sep = ",", header = FALSE)
            rowdata <- as.matrix(rowdata)
        
            # Determine if the raw data need to transpose
            if(isRowCount=="FALSE"){
                # Transpose the matrix
                rowdata <- t(rowdata)
            }
        
            # Delete the rows containing remove names
            removeNames <- c("alpha.contaminated", "beta.contaminated", "delta.contaminated", "Excluded", "gamma.contaminated", "miss", "NA","not applicable", "unclassified", "unknown", "Unknown", "zothers")
            for (name in removeNames) {
                rowdata <- rowdata[!grepl(name, rowdata[, 1]), ]
            }
        
            # Discard first row and first column (row name, column name)
            rowdata <- rowdata[-1, -1]
        
            # Convert character data in the matrix to double data using the mode() method
            mode(rowdata) <- "double"
        
            # Check whether NA is still present in the matrix
            any(is.na(rowdata))
        
            # Transpose the matrix, ready for use in the normalization method
            rowdata <- t(rowdata)
        
            # Run different normalize methods accroding to input param
            if (normname == "linnorm") {
                rowdata <- Linnorm(rowdata, minNonZeroPortion = 0.2)
            } else if (normname == "scran") {
                rowdata <- SCRAN_FN(rowdata)
            } else if (normname == "tmm") {
                rowdata <- TMM_FN(rowdata)
            } else if (normname == "scone") {
                rowdata <- DESEQ_FN(rowdata)
            } else if (normname == "cpm") {
                rowdata <- cpm(rowdata, log=FALSE)
            } else if (normname == "seurat") {
                rowdata <- as.matrix(NormalizeData(rowdata))
            } 

            return(rowdata)
        }
    """)

    robjects.r("""
        res = processDataset(path, isRowCount, normname)
    """)

    dataframe = robjects.reval("res")
    # normalized_dataset = np.array(dataframe)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        normalized_dataset = pd.DataFrame(data=robjects.conversion.rpy2py(dataframe))
        normalized_dataset.columns = normalized_dataset.iloc[0]
        normalized_dataset.drop(normalized_dataset.index[0])

    # Add columns name to dataframe
    normalized_dataset.columns = pd.read_csv(filepathpy, header=0, index_col=0).index

    # Return processed dataset
    return normalized_dataset

# Test
# filepath = '../originalDatasets/' + 'yan-RowCount.csv'
# normalized_dataset = dataCleanAndNormalize(filepath, True, "linnorm")
# print(normalized_dataset)
