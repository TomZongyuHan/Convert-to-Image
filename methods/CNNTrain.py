# Import library and methods


# Transfer non-image data to image dataset
# Input:
#   enhancedDataset: the variable of enhanced image datasets
#   CNNName: the name of CNN method
# Output:
#   result: the variable of CNN train result
def CNNTrain(enhancedDataset, CNNName):
    # Implement and run CNN methods
    if CNNName == 'alexnet':
        result = 1
    elif CNNName == 'vgg':
        result = 1
    elif CNNName == 'squeezenet':
        result = 1
    elif CNNName == 'resnet':
        result = 1
    elif CNNName == 'densenet':
        result = 1
    else:
        print("Please enter a correct CNN method name.")

    return result
