# Import libraries
library(Linnorm)
library(scone) # for scone/TTM/scran
library(edgeR) # for cpm
library(Seurat)

# This function used to read, clean and normalize data
# Input:
#   path: data csv file path
#   normName: normalization method name, should be linnorm/scone/ttm/scran/cpm/seurat
# Return:
#   processed data: rows - genes, columns - cells
processDataset <- function(path, normname) {
  # Read the CSV file and convert it to matrix format
  rowdata <- read.table(path, sep = ",", header = FALSE)
  rowdata <- as.matrix(rowdata)

  # Delete the rows containing remove names
  removeNames <- c("alpha.contaminated", "beta.contaminated", "delta.contaminated", "Excluded", "gamma.contaminated", "miss", "NA","not applicable", "unclassified", "unknown", "Unknown", "zothers")
  for (name in removeNames) {
    rowdata <- rowdata[!grepl(name, rowdata[, 1]), ]
  }

  # Save colume name and row name
  cn <- rowdata[, 1]
  rn <- rowdata[1, ]
  rn_c <- ""
  for (name in rn[-1]) {
    rn_c <- c(rn_c, name)
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
  
  # Add the rowname and colname of dataset
  rownames(rowdata) <- rn[-1]
  colnames(rowdata) <- cn[-1]
  
  # Output csv file
  filename <- paste(substring(path, 1, nchar(path) - 12), normname, ".csv", sep = "")
  write.table(rowdata, filename, sep = ",", col.names = NA)
}

# Run normalization - Eden
processDataset("yan-RowCount.csv", "linnorm")
processDataset("yan-RowCount.csv", "scran")
processDataset("yan-RowCount.csv", "tmm")
processDataset("yan-RowCount.csv", "scone")
processDataset("yan-RowCount.csv", "cpm")
processDataset("yan-RowCount.csv", "seurat")
processDataset("zeisel-RowCount.csv", "linnorm")
processDataset("zeisel-RowCount.csv", "scran")
processDataset("zeisel-RowCount.csv", "tmm")
processDataset("zeisel-RowCount.csv", "scone")
processDataset("zeisel-RowCount.csv", "cpm")
processDataset("zeisel-RowCount.csv", "seurat")

# Run normalization - Bruce
processDataset("TabulaMuris_Heart_10X-RowCount.csv", "linnorm")
processDataset("TabulaMuris_Heart_10X-RowCount.csv", "scran")
processDataset("TabulaMuris_Heart_10X-RowCount.csv", "tmm")
processDataset("TabulaMuris_Heart_10X-RowCount.csv", "scone")
processDataset("TabulaMuris_Heart_10X-RowCount.csv", "cpm")
processDataset("TabulaMuris_Heart_10X-RowCount.csv", "seurat")
processDataset("TabulaMuris_Liver_10X-RowCount.csv", "linnorm")
processDataset("TabulaMuris_Liver_10X-RowCount.csv", "scran")
processDataset("TabulaMuris_Liver_10X-RowCount.csv", "tmm")
processDataset("TabulaMuris_Liver_10X-RowCount.csv", "scone")
processDataset("TabulaMuris_Liver_10X-RowCount.csv", "cpm")
processDataset("TabulaMuris_Liver_10X-RowCount.csv", "seurat")

# Run normalization - Ariel
processDataset("xin-RowCount.csv", "linnorm")
processDataset("xin-RowCount.csv", "scran")
processDataset("xin-RowCount.csv", "tmm")
processDataset("xin-RowCount.csv", "scone")
processDataset("xin-RowCount.csv", "cpm")
processDataset("xin-RowCount.csv", "seurat")
processDataset("tasic-rpkms-RowCount.csv", "linnorm")
processDataset("tasic-rpkms-RowCount.csv", "scran")
processDataset("tasic-rpkms-RowCount.csv", "tmm")
processDataset("tasic-rpkms-RowCount.csv", "scone")
processDataset("tasic-rpkms-RowCount.csv", "cpm")
processDataset("tasic-rpkms-RowCount.csv", "seurat")

# Run normalization - Tom
processDataset("TabulaMuris_Trachea_FACS-RowCount.csv", "linnorm")
processDataset("TabulaMuris_Trachea_FACS-RowCount.csv", "scran")
processDataset("TabulaMuris_Trachea_FACS-RowCount.csv", "tmm")
processDataset("TabulaMuris_Trachea_FACS-RowCount.csv", "scone")
processDataset("TabulaMuris_Trachea_FACS-RowCount.csv", "cpm")
processDataset("TabulaMuris_Trachea_FACS-RowCount.csv", "seurat")
processDataset("TabulaMuris_Thymus_10X-RowCount.csv", "linnorm")
processDataset("TabulaMuris_Thymus_10X-RowCount.csv", "scran")
processDataset("TabulaMuris_Thymus_10X-RowCount.csv", "tmm")
processDataset("TabulaMuris_Thymus_10X-RowCount.csv", "scone")
processDataset("TabulaMuris_Thymus_10X-RowCount.csv", "cpm")
processDataset("TabulaMuris_Thymus_10X-RowCount.csv", "seurat")

# Run normalization - Brooks
processDataset("TabulaMuris_Marrow_10X-RowCount.csv", "linnorm")
processDataset("TabulaMuris_Marrow_10X-RowCount.csv", "scran")
processDataset("TabulaMuris_Marrow_10X-RowCount.csv", "tmm")
processDataset("TabulaMuris_Marrow_10X-RowCount.csv", "scone")
processDataset("TabulaMuris_Marrow_10X-RowCount.csv", "cpm")
processDataset("TabulaMuris_Marrow_10X-RowCount.csv", "seurat")
processDataset("TabulaMuris_Marrow_FACS-RowCount.csv", "linnorm")
processDataset("TabulaMuris_Marrow_FACS-RowCount.csv", "scran")
processDataset("TabulaMuris_Marrow_FACS-RowCount.csv", "tmm")
processDataset("TabulaMuris_Marrow_FACS-RowCount.csv", "scone")
processDataset("TabulaMuris_Marrow_FACS-RowCount.csv", "cpm")
processDataset("TabulaMuris_Marrow_FACS-RowCount.csv", "seurat")
