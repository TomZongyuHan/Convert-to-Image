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
    print("Image augumenting......")
    # Implement and run image enhance method
    # create two lists to store new img data
    newXTrainDataset = []
    newXTestDataset = imageDataset[1]
    newYTrainDataset = imageDataset[2].tolist()
    newYTestDataset = imageDataset[3]

    for img in imageDataset[0]:
        # 3d img -> 2d img
        # if img.ndim == 3:
        #     img = img[:, :, 0]
        # 3 128 128
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
    print(augmentedDataset[0].shape)
    return augmentedDataset

# Use random methods to enhance image data
# Input:
#   npImage: numpy format -> (128, 128)
# Output:
#   npImage: augmented data -> (128, 128)
def random_aug(npImage):
    # generate a random int
    op = np.random.randint(1, 7)
    # crop
    if (op == 1):
        npImage = crop(npImage, 100, 100)
    # random flip
    if (op == 2):
        npImage = random_flip(npImage)
    # zoom
    if (op == 3):
        npImage = zoom(npImage, 100, 100, 2)
    # gauss_noise
    if (op == 4):
        npImage = gasuss_noise(npImage, 0, 0.005)
    # brightness
    if (op == 5):
        npImage = change_brightness(npImage)
    # shift
    if (op == 6):
        npImage = shift(npImage)
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
    # npImage = cv2.resize(npImage, (int(height * 1.5), int(width * 1.5)))
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

# Add a noise layer to the picture
# Input:
#   npImage: numpy format -> (128, 128)
#   mean: The mean
#   var: The variance
# Output:
#   out: augmented data -> (128, 128)
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

# change the brightness of numpy image
# Input:
#   npImage: numpy format -> (128, 128)
# Output:
#   npImage: augmented data -> (128, 128)
def change_brightness(npImage):
    npImage = tf.image.random_brightness(npImage, 0.5)
    return npImage

# Image translation, Shift it 28 pixels to the right and 28 pixels down
# Input:
#   npImage: numpy format -> (128, 128)
# Output:
#   npImage: augmented data -> (128, 128)
def shift(npImage):
    npImage = npImage.transpose(1, 2, 0)
    npImage = Image.fromarray(np.uint8(npImage))
    npImage = ImageChops.offset(npImage, 28, 28)
    npImage = np.array(npImage)
    npImage = npImage.transpose(2, 0, 1)
    return npImage

# Test
# imageDataset = np.load("../testnpy.npy", allow_pickle=True)
# ([x train], [x test], [y train], [y test])
# print(imageDataset[0].shape)
# [[144, 3, 128, 128], [18, 3, 128, 128], [144,], [18]]
# augmented_dataset = imageAugumentation(imageDataset)
# print(augmented_dataset[1].shape)
# npImage = npImage.transpose(0, 2, 3, 1) -> [144, 128, 128, 3]