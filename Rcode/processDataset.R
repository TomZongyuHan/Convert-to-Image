# Import libraries
library(Linnorm) # for linnorm
library(scone) # for scone/TTM/scran
library(edgeR) # for cpm
library(Seurat) # for seurat

# This function used to read, clean and normalize data
# Input:
#   path: data csv file path
#   normName: normalization method name, should be linnorm/scone/ttm/scran/cpm/seurat
# Return:
#   processed data: rows - genes, columns - cells
processDataset <- function(path, isRowCount) {
  # Read the CSV file and convert it to matrix format
  rowdata <- read.table(path, sep = ",", header = FALSE)
  rowdata <- as.matrix(rowdata)

  # Determine if the raw data need to transpose
  if(!isRowCount){
    # Transpose the matrix
    rowdata <- t(rowdata)
  }

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
  # linnorm
  rowdata_linnorm <- Linnorm(rowdata, minNonZeroPortion = 0.2)
  #scran
  rowdata_scran <- SCRAN_FN(rowdata)
  #tmm
  rowdata_tmm <- TMM_FN(rowdata)
  #scone
  rowdata_scone <- DESEQ_FN(rowdata)
  #cpm
  rowdata_cpm <- cpm(rowdata, log=FALSE)
  #seurat
  rowdata_seurat <- as.matrix(NormalizeData(rowdata))

  # Add the rowname and colname of dataset
  rownames(rowdata_linnorm) <- rn[-1]
  colnames(rowdata_linnorm) <- cn[-1]
  rownames(rowdata_scran) <- rn[-1]
  colnames(rowdata_scran) <- cn[-1]
  rownames(rowdata_tmm) <- rn[-1]
  colnames(rowdata_tmm) <- cn[-1]
  rownames(rowdata_scone) <- rn[-1]
  colnames(rowdata_scone) <- cn[-1]
  rownames(rowdata_cpm) <- rn[-1]
  colnames(rowdata_cpm) <- cn[-1]
  rownames(rowdata_seurat) <- rn[-1]
  colnames(rowdata_seurat) <- cn[-1]
  
  # Output csv file
  filename_linnorm <- paste(substring(path, 1, nchar(path) - 12), "linnorm.csv", sep = "")
  filename_scran <- paste(substring(path, 1, nchar(path) - 12), "scran.csv", sep = "")
  filename_tmm <- paste(substring(path, 1, nchar(path) - 12), "tmm.csv", sep = "")
  filename_scone <- paste(substring(path, 1, nchar(path) - 12), "scone.csv", sep = "")
  filename_cpm <- paste(substring(path, 1, nchar(path) - 12), "cpm.csv", sep = "")
  filename_seurat <- paste(substring(path, 1, nchar(path) - 12), "seurat.csv", sep = "")

  write.table(rowdata_linnorm, filename_linnorm, sep = ",", col.names = NA)
  write.table(rowdata_scran, filename_scran, sep = ",", col.names = NA)
  write.table(rowdata_tmm, filename_tmm, sep = ",", col.names = NA)
  write.table(rowdata_scone, filename_scone, sep = ",", col.names = NA)
  write.table(rowdata_cpm, filename_cpm, sep = ",", col.names = NA)
  write.table(rowdata_seurat, filename_seurat, sep = ",", col.names = NA)
}

# Run normalization
processDataset("TabulaMuris_Heart_10X-RowCount.csv", TRUE)
processDataset("TabulaMuris_Liver_10X-RowCount.csv", TRUE)
processDataset("TabulaMuris_Marrow_10X-RowCount.csv", TRUE)
processDataset("TabulaMuris_Marrow_FACS-RowCount.csv", TRUE)
processDataset("TabulaMuris_Thymus_10X-RowCount.csv", TRUE)
processDataset("TabulaMuris_Trachea_FACS-RowCount.csv", TRUE)
processDataset("tasic-rpkms-RowCount.csv", TRUE)
processDataset("xin-RowCount.csv", TRUE)
processDataset("yan-RowCount.csv", TRUE)
processDataset("zeisel-RowCount.csv", TRUE)
processDataset("baron-mouse-RawCount.csv", FALSE)
processDataset("deng-reads-RawCount.csv", FALSE)
processDataset("manno_human-RawCount.csv", FALSE)
processDataset("manno_mouse-RawCount.csv", FALSE)
processDataset("TabulaMuris_Heart_10X-RawCount.csv", FALSE)
processDataset("TabulaMuris_Heart_FACS-RawCount.csv", FALSE)
processDataset("zhengmix4uneq-RawCount.csv", FALSE)