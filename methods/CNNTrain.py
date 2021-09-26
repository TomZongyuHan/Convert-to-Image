# Import library and methods
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
from tqdm import tqdm


# Transfer non-image data to image dataset
# Input:
#   enhancedDataset: the variable of enhanced image datasets
#   CNNName: the name of CNN method
# Output:
#   result: the list of CNN train result
def CNNTrain(augmentedDataset, CNNName):
    # print("Training......")
    # Ignore warnings and set import config 
    warnings.simplefilter('ignore')
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    # Check if can use cuda to run cnn train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and get number of classes
    num_classes, trainloader, testloader = loadDataset(augmentedDataset, CNNName, device)

    # Import specific model online and put into device
    model = getModel(CNNName, num_classes).to(device)

    # Train cnnModel
    model = trainModel(model, trainloader)

    # Run model to evaluate test data
    testResult, testLabel = evaluateData(model, testloader)

    trainResult, trainLable = evaluateData(model, trainloader)
    train_accuracy = accuracy_score(trainResult, trainLable)
    # print('!!!!!!!!!!!!!!!!')
    # print(train_accuracy)
    # print('!!!!!!!!!!!!!!!!')

    return [testResult, testLabel, train_accuracy]


def loadDataset(augmentedDataset, CNNName, device):
    # Get all data need to be use
    X_train_img = augmentedDataset[0].transpose(0, 2, 3, 1)
    X_test_img = augmentedDataset[1].transpose(0, 2, 3, 1)
    y_train = augmentedDataset[2]
    y_test = augmentedDataset[3]

    # Set preprocess transforms
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])

    # Encode the y train dataset
    le = LabelEncoder()
    num_classes = np.unique(le.fit_transform(y_train)).size

    # Preprocess the dataset to tensor float type and transfer it into device
    X_train_img = torch.stack([preprocess(img) for img in X_train_img]).float().to(device)
    X_test_img = torch.stack([preprocess(img) for img in X_test_img]).float().to(device)
    y_train = torch.from_numpy(le.fit_transform(y_train)).to(device)
    y_test = torch.from_numpy(le.transform(y_test)).to(device)

    # Set the batch size and put dataset into dataloader
    batch_size = 256
    trainloader = DataLoader(TensorDataset(X_train_img, y_train), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(TensorDataset(X_test_img, y_test), batch_size=batch_size, shuffle=True)

    return num_classes, trainloader, testloader


def getModel(CNNName, num_classes):
    modelPath = 'cnnModels/' + CNNName + '.pt'
    if os.path.exists(modelPath):
        model = torch.load(modelPath)
    else:
        if CNNName == 'alexnet':
            model = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=False, verbose=False)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif CNNName == 'vgg16':
            model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=False, verbose=False)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif CNNName == 'squeezenet':
            model = torch.hub.load('pytorch/vision:v0.9.0', 'squeezenet1_1', pretrained=False, verbose=False)
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        elif CNNName == 'resnet':
            model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False, verbose=False)
            model.fc = nn.Linear(2048, num_classes)
        elif CNNName == 'densenet':
            model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=False, verbose=False)
            model.fc = nn.Linear(18085, num_classes)
        torch.save(model, modelPath)
    
    return model


def trainModel(model, trainloader):
    # Set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    # Run training use the epoch num
    accuracy = 0
    epochNum = 600
    for epoch in tqdm(range(epochNum)):
    #while True:
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #     predicted = torch.argmax(outputs, 1)
        #     epoch += 1
        #     accuracy = accuracy_score(predicted.cpu().numpy(), labels.cpu().numpy())
        #     print(str(epoch) + ': ' + str(accuracy))
        # if accuracy > 0.95:
        #     break
    
    # print(accuracy)
        # # print epoch statistics
        # print('[%d] loss: %.3f' %
        #       (epoch + 1, running_loss / len(X_train_tensor) * batch_size))
    
    return model


def evaluateData(model, testloader):
    model.eval()
    outputs = []
    with torch.no_grad():
        results = []
        trueLabels = []
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            # forward + backward + optimize
            outputs = model(inputs)
            test_predicted = torch.argmax(outputs, 1)
            results.append(test_predicted.cpu().numpy())
            trueLabels.append(labels.cpu().numpy())
    testResult = [b for a in results for b in a]
    testLabel = [b for a in trueLabels for b in a]

    return testResult, testLabel


# # test CNNTrain
# x = CNNTrain(1, "vgg16")
# x = np.array(x)
# np.save('test_Ariel.npy', x)
