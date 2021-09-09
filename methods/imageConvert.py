# Import library and methods
from sklearn.model_selection import train_test_split
from pyDeepInsight import ImageTransformer


# Transfer non-image data to image dataset
# Input:
#   dataset: the variable of datasets input
#       this dataset will be normalized dataset if use drMethod
#   drMethod: the method of dimentionaliry reduction
#       if do not use it, just transfer two params
#   icName: the name of image convert method
# Output:
#   imageDataset: the list of image datasets
#       imageDataset[0] is X train data
#       imageDataset[1] is X test data
def imageConvert(drResult, icName):
    if isinstance(drResult, list):
        drMethod = drResult[0]
        dataset = drResult[1]
    else:
        dataset = drResult

    # Implement and run image convert methods
    if icName == 'deepinsight':
        # Divide the dataset
        x = dataset.values.transpose()
        y = dataset.columns.values
        print(x.shape)
        print(y.shape)
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=23, stratify=y)

        # Implement method, Deepinsight need given drmethod
        it = ImageTransformer(feature_extractor=drMethod, 
            pixels=50, random_state=1701, 
            n_jobs=-1)
        
        # Train and get image data
        X_train_img = it.fit_transform(X_train)
        X_test_img = it.fit_transform(X_test)
        imageDataset = [X_train_img, X_test_img]
    elif icName == 'cpcr':
        imageDataset = 1
    elif icName == 'gaf':
        imageDataset = 1
    else:
        print("????? Please enter a correct image convert method name ?????")

    return imageDataset
