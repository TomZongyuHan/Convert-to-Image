# Used to install all python and R dependencies
# Import library and methods
import os


# Install python dependencies
def installPythonDeps():
    # Set all package name or address
    packageNames = [
        'rpy2', 
        'numpy', 
        'scikit-learn', 
        'pandas',
        'matplotlib', 
        'git+git://github.com/alok-ai-lab/DeepInsight.git#egg=DeepInsight',
        'phate',
        'pyts',
        'tensorflow',
        'torch',
        'torchvision',
        'scikit-image',
        'tqdm',
        'umap-learn'
    ]

    # For loop to install all packages
    for packageName in packageNames:
        os.system("pip install " + packageName)


# Install R dependencies
def installRDeps():
    # Import packages after install
    import rpy2.robjects as robjects

    # Call method in package to install dependencies
    # Install R's packages
    robjects.r("""
        install.packages("BiocManager", repos = "http://cran.r-project.org")
        BiocManager::install("Linnorm", force = TRUE)
        BiocManager::install("edgeR", force = TRUE)
        BiocManager::install("scone", force = TRUE)
        BiocManager::install("scran", force = TRUE)
        install.packages("Seurat", repos = "http://cran.r-project.org")
    """)


# Main function entry
if __name__ == '__main__':
    installPythonDeps()
    installRDeps()
    print('!!!!! All packages install complete !!!')