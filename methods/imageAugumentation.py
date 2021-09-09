# Import library and methods
import unittest
from PIL import Image, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
import cv2

# Enhance the image dataset
# Input:
#   imageDataset: the variable of datasets after dimensionality reduce
# Output:
#   enhancedDataset: the list of test set and train set
def imageAugumentation(imageDataset):
    # Implement and run image enhance method
    augmentedDataset = []
    img = imageDataset[0]
    height, width, _ = img.shape
    # op = np.random.randint(1, 6)
    op = 3
    # crop
    if(op == 1):
        img = crop(img, 1512, 2016)
    # random flip
    if(op == 2):
        img = random_flip(img)
    # zoom
    if (op == 3):
        img = zoom(img, 1512, 2016, 2)
    # gauss_noise
    if (op == 4):
        img = gasuss_noise(img, 0, 0.005)
    # adjust contrast
    if (op == 5):
        img = adjust_contrast(img)

    img = Image.fromarray(img)
    augmentedDataset.append(img)
    return augmentedDataset

def crop(npImage, height_range, width_range): #1512, 2016
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = npImage.shape
    new_height = np.random.randint(0, height - height_range)
    new_width = np.random.randint(0, width - width_range)
    npImage = npImage[new_height: new_height + height_range, new_width: new_width + width_range, :]
    return npImage

def zoom(npImage, height_range, width_range, magnification):
    height, width, _ = npImage.shape
    # npImage = cv2.resize(npImage, (int(height * 1.5), int(width * 1.5)))
    npImage = npImage[int((height - height_range) / magnification): int((height + height_range) / magnification),
          int((width - width_range) / magnification): int((width + width_range) / magnification), :]
    return npImage

def random_flip(npImage):
    flip_op = np.random.randint(1, 4)
    # flip vertical
    if(flip_op == 1):
        npImage = np.flip(npImage)
    # flip horizontal
    if(flip_op == 2):
        npImage = np.fliplr(npImage)
    # rotate 90 degree
    if(flip_op == 3):
        npImage = np.rot90(npImage)
    return npImage

def gasuss_noise(npImage, mean = 0, var = 0.005):
    # nomorlized the pixel value
    npImage = np.array(npImage / 255, dtype = float)
    # create a gasuss matrix
    noise = np.random.normal(mean, var ** 0.5, npImage.shape)
    # combine the image matrix with the gasuss matrix
    out = npImage + noise
    # Restores to the original pixel value
    out = np.uint8(out * 255)
    return out

def adjust_contrast(npImage):
    random_contrast_factor = np.random.randint(10, 31) / 10
    npImage = Image.fromarray(npImage)
    npImage = ImageEnhance.Contrast(npImage).enhance(random_contrast_factor)
    npImage = np.array(npImage)
    return npImage

# Test
imagePath = '../originalDatasets/' + 'testImage.jpg'
img = Image.open(imagePath)
img = np.array(img)
imageDataset = []
imageDataset.append(img)
augmented_dataset = imageAugumentation(imageDataset)
plt.imshow(augmented_dataset[0])
plt.show()