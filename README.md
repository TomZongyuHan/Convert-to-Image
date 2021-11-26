<!--
 * @Description: 
 * @Author: Mo Xu
 * @Date: 2021-09-10 01:29:22
 * @LastEditors: Mo Xu
 * @LastEditTime: 2021-11-26 16:18:19
 * @FilePath: /27-1/README.md
-->
# Converting Non-Image Data to Image Data in Single Cell Data
![image](https://img.shields.io/badge/Type-Algorithm-important)
![image](https://img.shields.io/badge/Language-Python-blue)
![image](https://img.shields.io/badge/Version-0.0.1-green)


## Background
Convolutional neural networks (CNNs) are a class of deep learning architectures that have shown promising results and gained widespread attention concerning image data. CNN takes an input image (e.g., p × q feature matrix) and through its hidden layers conducts feature extraction and classification. One of the key advantages of CNN is their high efficiency, i.e. fewer samples and less training time are needed to achieve good levels of performance. This led to their high popularity in a myriad of cutting-edge commercial applications. As input, CNN takes an image. In a local region, an image is comprised of spatially coherent pixels, i.e., similar information is shared by the pixels near each other. If pixels are arbitrarily arranged, then their placement can negatively impact the performance of the feature extraction and classification process. Consequently, the order of the nearby pixels of an image in CNN is not independent. Thus, converting non image data to image data considering spatially coherent pixels in local regions plays an important role in the non-image data transformation. Hence, the transformation helps to improve the resolution of cell types in the single cell data.

> This project is the capstone for comp5703 in USYD in 2021 Semeter 2


## Install
### If you run in local
1. Python install
   - Please refer to [python website](https://www.python.org/)
   - Note: Please use python3 for this project
2. R install
   - Please refer to [R website](https://www.r-project.org/)
3. Dependencies install
   - Open commond line, enter the project folder, run command: `python3 installDependencies.py`

### If you run in Colab
1. Upload files
   - Upload run_colab.ipynb to Colab as a new notebook
   - Upload the whole project folder named "27-1" to Google Drive root directory
2. Import files and dependencies install
   - Run first block and import Google Drive 
   - Run second block to install all dependencies 
   - Note: Halfway through the installation of dependencies, you need to enter command to confirm whether to update all dependencies. It is recommended to enter `a` to update all dependencies.
   - Note: You need to import Google Drive and install dependencies every time open a new Colab session

## Run
### If you run in local
1. Run test 
   - Open commond line, enter the project folder, run command: `python3 test.py`
   - Note: you only need to run test at first time use this pipeline
2. Run main
   - Put your dataset file end with .csv in originalDatasets
   - Modify last two lines about file name and if row count True/False setting
   - Open commond line, enter the project folder, run command: `python3 main.py`
   - Note: If you stop running for some reason when you run the pipeline halfway, the next time it runs, it will continue to run according to the latest progress of the current data set
  

### If you run in Colab
1. Run test 
   - Run third block in notebook to have a quick check(but maybe take hours) of the pipeline
   - Note: you only need to run test at first time use this pipeline
2. Run main
   - Put your dataset file end with .csv in originalDatasets
   - Modify last two lines about file name and if row count True/False setting
   - Run fourth block in notebook to run main pipeline
   - Note: If you stop running for some reason when you run the pipeline halfway, the next time it runs, it will continue to run according to the latest progress of the current data set


### Get results
- After finish the pipeline running, all results collected in results.csv file in results/accuracies
- Note: When you want to run another dataset, you need to **delete** all model files in cnnModels and two result files in results/accuracies


## Technical Components
1. __Python__: as main language
2. __R__: as normalization function language
3. __Machine Learning__: as main technology


## License
[MIT](LICENSE) © Eden © Tom © Brooks © Ariel © Bruce
