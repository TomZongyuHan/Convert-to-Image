# Import library and methods
from sklearn.model_selection import train_test_split
from pyDeepInsight import ImageTransformer, LogScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
import phate
from pyts.image import GramianAngularField
import numpy as np
import cv2
import umap


# Transfer non-image data to image dataset
# Input:
#   dataset: the variable of datasets input
#       this dataset will be normalized dataset if use drMethod
#   drMethod: the method of dimentionaliry reduction
#       if do not use it, just transfer two params
#   icName: the name of image convert method
# Output:
#   imageDataset: the list of image datasets
#       imageDataset[0] is X train data: (imgNum, channels, pixel, pixel)
#       imageDataset[1] is X test data: (imgNum, channels, pixel, pixel)
#       imageDataset[2] is y train data
#       imageDataset[3] is y test data
def imageConvert(drResult, icName):
    # Run image convert methods
    if icName == 'deepinsight':
        drName = drResult[0]
        dataset = drResult[1]
        if drName == 'pca':
            drMethod = PCA(n_components=2)
        elif drName == 'kpca':
            drMethod = KernelPCA(n_components=2, kernel='cosine', n_jobs=-1)
        elif drName == 'tsne':
            drMethod = TSNE(n_components=2, n_jobs=-1)
        elif drName == 'phate':
            drMethod = phate.PHATE(n_components=2, n_jobs=-1)
        elif drName == 'umap':
            drMethod = umap.UMAP(n_components=2)
        imageDataset = deepinsight(drMethod, dataset)
    elif icName == 'cpcr':
        dataset = drResult[2]
        labels = drResult[3]
        imageDataset = cpcr(dataset, labels)
    elif icName == 'gaf':
        dataset = drResult[2]
        labels = drResult[3]
        imageDataset = gaf(dataset, labels)
    else:
        print("????? Please enter a correct image convert method name ?????")

    return imageDataset


# Function to run deepinsight method and return image results
# Input: 
#   drMethod: dimentionaliry method
#   dataset: the dataset after clean and normalize but not dr
# Output:
#   Refer to imageConvert()
def deepinsight(drMethod, dataset):
    # Divide dataset
    datas = dataset.values.transpose()
    labels = dataset.columns.values
    X_train, X_test, y_train, y_test = train_test_split(
        datas, labels, test_size=0.2, random_state=23, stratify=labels)

    # Normalize the data for deepinsight requirement
    ln = LogScaler()
    X_train_norm = ln.fit_transform(X_train)
    X_test_norm = ln.transform(X_test)

    # Implement method, Deepinsight need given drmethod
    it = ImageTransformer(feature_extractor=drMethod,
        pixels=128, random_state=1701)

    # Train and get image data
    X_train_img = it.fit_transform(X_train_norm).transpose(0, 3, 1, 2)
    X_test_img = it.transform(X_test_norm).transpose(0, 3, 1, 2)

    return [X_train_img, X_test_img, y_train, y_test]


# Function to run cpcr method and return image results
# Input: 
#   dataset: the dataset after clean and normalize but not dr
#   labels: list of labels of the dataset
# Output:
#   Refer to imageConvert()
def cpcr(dataset, labels):
    dataset = np.array(dataset)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 127), copy = False)
    dataset = min_max_scaler.fit_transform(dataset).T

    imgs = []
    for data in dataset:
        pairs = []
        dataSize = len(data)
        terms = dataSize // 3
        for i in range(0, terms):
            grey_intensity_0 = int(i * 255 / terms)
            grey_intensity_1 = int((i+0.5) * 255 / terms)
            pair_0 = [int(data[3 * i]), int(data[3 * i + 1]), grey_intensity_0]
            pair_1 = [int(data[3 * i + 2]), int(data[3 * i + 2]), grey_intensity_1]
            pairs.append(pair_0)
            pairs.append(pair_1)
        
        img = np.zeros([128, 128], dtype = int)
        for pair in pairs:
            x = pair[0]
            y = pair[1]
            grey_intensity = pair[2]
            img[x][y] += grey_intensity
            if img[x][y] > 255:
                img[x][y] = 255
        
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB).T
        imgs.append(img)
    
    imgs = np.array(imgs)

    X_train, X_test, y_train, y_test = train_test_split(
        imgs, labels, test_size=0.2, random_state=23, stratify=labels)

    return [X_train, X_test, y_train, y_test]


# Function to run gaf method and return image results
# Input: 
#   dataset: the dataset after clean and normalize but not dr
#   labels: list of labels of the dataset
# Output:
#   Refer to imageConvert()
def gaf(dataset, labels):
    # Implement and fit transform dataset
    dataset = np.array(dataset).T
    # If number of features of dataset is smaller than 128, it cannot be transfer to 128x128 image in gaf, so we should handle it
    shape = dataset.shape
    if shape[1] < 128:
        # Implement the gaf method and tranform the data to image
        gaf = GramianAngularField(image_size=shape[1])
        imgs = np.array(gaf.fit_transform(dataset))
        new_imgs = []
        # All images should be strtched to 128 pixels
        for img in imgs:
            new_img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
            new_img = cv2.cvtColor(np.float32(new_img), cv2.COLOR_GRAY2RGB).T
            new_imgs.append(new_img)
        imgs = np.array(new_imgs)
    else:
        gaf = GramianAngularField(image_size=128)
        imgs = np.array(gaf.fit_transform(dataset))
        new_imgs = []
        for img in imgs:
            new_img = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB).T
            new_imgs.append(new_img)
        imgs = np.array(new_imgs)

    # Split and return image result
    X_train, X_test, y_train, y_test = train_test_split(
        imgs, labels, test_size=0.2, random_state=23, stratify=labels)
    
    return [X_train, X_test, y_train, y_test]
