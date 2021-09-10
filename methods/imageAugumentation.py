# Import library and methods
import cv2
from PIL import Image, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt

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
    newYTrainDataset = imageDataset[2].tolist()
    newYTestDataset = imageDataset[3]

    for img in imageDataset[0]:
        # 3d img -> 2d img
        # img = img[:, :, 0]
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
#   npImage: numpy format -> (50, 50)
# Output:
#   npImage: augmented data -> (50, 50)
def random_aug(npImage):
    # generate a random int
    op = np.random.randint(1, 5)
    # crop
    if (op == 1):
        npImage = crop(npImage, 40, 40)
    # random flip
    if (op == 2):
        npImage = random_flip(npImage)
    # zoom
    if (op == 3):
        npImage = zoom(npImage, 40, 40, 2)
    # gauss_noise
    if (op == 4):
        npImage = gasuss_noise(npImage, 0, 0.005)
    # adjust contrast
    if (op == 5):
        npImage = adjust_contrast(npImage)

    npImage = cv2.resize(npImage, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
    npImage = np.array(npImage)

    return npImage

# Crop the picture
# Input:
#   npImage: numpy format -> (50, 50)
#   height_range: image height required
#   width_range: image width required
# Output:
#   npImage: augmented data -> (50, 50)
def crop(npImage, height_range, width_range):
    npImage = np.array(npImage)
    height, width, _ = npImage.shape
    new_height = np.random.randint(0, height - height_range)
    new_width = np.random.randint(0, width - width_range)
    npImage = npImage[new_height: new_height + height_range, new_width: new_width + width_range]

    return npImage

# Use zoom operation on the image
# Input:
#   npImage: numpy format -> (50, 50)
#   height_range: image height required
#   width_range: image width required
#   magnification: need to enlarge the image multiple
# Output:
#   npImage: augmented data -> (50, 50)
def zoom(npImage, height_range, width_range, magnification):
    height, width, _ = npImage.shape
    # npImage = cv2.resize(npImage, (int(height * 1.5), int(width * 1.5)))
    npImage = npImage[int((height - height_range) / magnification): int((height + height_range) / magnification),
              int((width - width_range) / magnification): int((width + width_range) / magnification)]
    return npImage

# Flip the picture randomly
# Input:
#   npImage: numpy format -> (50, 50)
# Output:
#   npImage: augmented data -> (50, 50)
def random_flip(npImage):
    flip_op = np.random.randint(1, 4)
    # flip vertical
    if (flip_op == 1):
        npImage = np.flip(npImage)
    # flip horizontal
    if (flip_op == 2):
        npImage = np.fliplr(npImage)
    # rotate 90 degree
    if (flip_op == 3):
        npImage = np.rot90(npImage)
    return npImage

# Add a noise layer to the picture
# Input:
#   npImage: numpy format -> (50, 50)
#   mean: The mean
#   var: The variance
# Output:
#   out: augmented data -> (50, 50)
def gasuss_noise(npImage, mean = 0, var = 0.005):
    # nomorlized the pixel value
    npImage = np.array(npImage / 255, dtype=float)
    # create a gasuss matrix
    noise = np.random.normal(mean, var ** 0.5, npImage.shape)
    # combine the image matrix with the gasuss matrix
    out = npImage + noise
    # Restores to the original pixel value
    out = np.uint8(out * 255)
    return out

# Adjust the contrast of the image
# Input:
#   npImage: numpy format -> (50, 50)
# Output:
#   npImage: augmented data -> (50, 50)
def adjust_contrast(npImage):
    npImage = Image.fromarray(npImage)
    npImage = ImageEnhance.Contrast(npImage)
    npImage = np.array(npImage)
    return npImage

# Test
# imageDataset = np.load("../test.npy", allow_pickle=True)
# ([x train], [x test], [y train], [y test])
# augmented_dataset = imageAugumentation(imageDataset)
# print(augmented_dataset[0].shape)