# Import library and methods
import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageChops, Image
from skimage import transform

# Enhance the image dataset
# Input:
#   imageDataset: the variable of datasets after dimensionality reduce
# Output:
#   augmentedDataset: the list of test set and train set
def imageAugumentation(imageDataset):
    # Implement and run image enhance method
    # create two lists to store new img data
    newXTrainDataset = []
    newXTestDataset = imageDataset[1]
    if not isinstance(imageDataset[2], list):
        newYTrainDataset = imageDataset[2].tolist()
    else:
        newYTrainDataset = imageDataset[2]
    newYTestDataset = imageDataset[3]

    for img in imageDataset[0]:
        newXTrainDataset.append(img)

    augmentedXTrainDataset = newXTrainDataset
    for i in range(len(newXTrainDataset)):
        # apply random_aug method to enhance the datasets
        new_train_img = random_aug(newXTrainDataset[i])
        augmentedXTrainDataset.append(new_train_img)
        newYTrainDataset.append(newYTrainDataset[i])

    newYTrainDataset = np.array(newYTrainDataset)

    augmentedXTrainDataset = np.array(augmentedXTrainDataset)

    # 0 -> x train dataset, 1 -> x test dataset, 2 -> y train dataset, 3 -> y test dataset
    augmentedDataset = [augmentedXTrainDataset, newXTestDataset, newYTrainDataset , newYTestDataset]
    return augmentedDataset

# Use random methods to enhance image data
# Input:
#   npImage: numpy format -> (128, 128)
# Output:
#   npImage: augmented data -> (128, 128)
def random_aug(npImage):
    # generate a random int
    op = np.random.randint(1, 5)
    # crop
    if (op == 1):
        npImage = crop(npImage, 100, 100)
    # random flip
    if (op == 2):
        npImage = random_flip(npImage)
    # zoom
    if (op == 3):
        npImage = zoom(npImage, 100, 100, 2)
    # brightness
    if (op == 4):
        npImage = change_brightness(npImage)
    return npImage

# Crop the picture
# Input:
#   npImage: numpy format -> (128, 128)
#   height_range: image height required
#   width_range: image width required
# Output:
#   npImage: augmented data -> (128, 128)
def crop(npImage, height_range, width_range):
    npImage = np.array(npImage)
    _, height, width = npImage.shape
    new_height = np.random.randint(0, height - height_range)
    new_width = np.random.randint(0, width - width_range)
    npImage = npImage[:, new_height: new_height + height_range, new_width: new_width + width_range]
    npImage = transform.resize(npImage, (3, 128, 128))
    npImage = np.array(npImage)

    return npImage

# Use zoom operation on the image
# Input:
#   npImage: numpy format -> (128, 128)
#   height_range: image height required
#   width_range: image width required
#   magnification: need to enlarge the image multiple
# Output:
#   npImage: augmented data -> (128, 128)
def zoom(npImage, height_range, width_range, magnification):
    _, height, width = npImage.shape
    npImage = npImage[:, int((height - height_range) / magnification): int((height + height_range) / magnification),
              int((width - width_range) / magnification): int((width + width_range) / magnification)]
    npImage = transform.resize(npImage, (3, 128, 128))
    npImage = np.array(npImage)
    return npImage

# Flip the picture randomly
# Input:
#   npImage: numpy format -> (128, 128)
# Output:
#   npImage: augmented data -> (128, 128)
def random_flip(npImage):
    flip_op = np.random.randint(1, 3)
    # flip vertical
    if (flip_op == 1):
        npImage = cv2.flip(npImage, 0)
    # flip horizontal
    if (flip_op == 2):
        npImage = np.flip(npImage, 1)
    return npImage

# change the brightness of numpy image
# Input:
#   npImage: numpy format -> (128, 128)
# Output:
#   npImage: augmented data -> (128, 128)
def change_brightness(npImage):
    npImage = tf.image.random_brightness(npImage, 0.5)
    return npImage