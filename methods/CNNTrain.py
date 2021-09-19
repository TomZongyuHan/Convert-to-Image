# Import library and methods
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from skimage import transform
import warnings

warnings.simplefilter('ignore')


# Transfer non-image data to image dataset
# Input:
#   enhancedDataset: the variable of enhanced image datasets
#   CNNName: the name of CNN method
# Output:
#   result: the variable of CNN train result
def CNNTrain(augmentedDataset, CNNName):
    print("Training......")
    # Implement and run CNN methods
    if CNNName == 'alexnet':

        X_train_img = augmentedDataset[0].transpose(0, 2, 3, 1)
        X_test_img = augmentedDataset[1].transpose(0, 2, 3, 1)
        y_train = augmentedDataset[2]
        y_test = augmentedDataset[3]

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        num_classes = np.unique(y_train_enc).size

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        net = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=False, verbose=False)
        # net.classifier[1] = nn.Linear(512, num_classes)
        net.fc = nn.Linear(4096, num_classes)

        net

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = net.to(device)

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224)
        ])

        X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float().to(device)
        y_train_tensor = torch.from_numpy(le.fit_transform(y_train)).to(device)

        X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float().to(device)
        y_test_tensor = torch.from_numpy(le.transform(y_test)).to(device)

        X_train_tensor.dtype

        batch_size = 20

        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

        for epoch in range(50):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            # # print epoch statistics
            # print('[%d] loss: %.3f' %
            #       (epoch + 1, running_loss / len(X_train_tensor) * batch_size))


        testset = TensorDataset(X_test_tensor, y_test_tensor)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
        # Run evaluation function to get results
        testResult, testLabel = evaluation(testloader, net)


    elif CNNName == 'vgg16':

        # augmentedDataset = np.load("./test_Bruce.npy", allow_pickle = True)
        X_train_img = augmentedDataset[0].transpose(0, 2, 3, 1)
        X_test_img = augmentedDataset[1].transpose(0, 2, 3, 1)
        y_train = augmentedDataset[2]
        y_test = augmentedDataset[3]

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        num_classes = np.unique(y_train_enc).size

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        net = torch.hub.load(
            'pytorch/vision:v0.9.0', 'vgg16', pretrained=False, verbose=False)
        net.classifier[6] = nn.Linear(4096, num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = net.to(device)

        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

        X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float().to(device)
        y_train_tensor = torch.from_numpy(le.fit_transform(y_train)).to(device)

        X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float().to(device)
        y_test_tensor = torch.from_numpy(le.transform(y_test)).to(device)

        batch_size = 10

        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

        # 50 epoch variance=0
        for epoch in range(50):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            # print epoch
            # print('[%d] loss: %.3f' %
            # (epoch + 1, running_loss / len(X_train_tensor) * batch_size))

        testset = TensorDataset(X_test_tensor, y_test_tensor)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        # Run evaluation function to get results
        testResult, testLabel = evaluation(testloader, net)


    elif CNNName == 'squeezenet':
        X_train_img = augmentedDataset[0].transpose(0, 2, 3, 1)
        X_test_img = augmentedDataset[1].transpose(0, 2, 3, 1)
        y_train = augmentedDataset[2]
        y_test = augmentedDataset[3]

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        num_classes = np.unique(y_train_enc).size

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        net = torch.hub.load('pytorch/vision:v0.9.0', 'squeezenet1_1', pretrained=False, verbose=False)
        net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = net.to(device)

        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

        X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float().to(device)
        y_train_tensor = torch.from_numpy(le.fit_transform(y_train)).to(device)

        X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float().to(device)
        y_test_tensor = torch.from_numpy(le.transform(y_test)).to(device)

        X_train_tensor.dtype

        batch_size = 10

        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

        # 50epoch batch_size=10 variance=0 annotation=yes
        for epoch in range(50):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            # # print epoch statistics
            # print('[%d] loss: %.3f' %
            #       (epoch + 1, running_loss / len(X_train_tensor) * batch_size))


        testset = TensorDataset(X_test_tensor, y_test_tensor)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
        # Run evaluation function to get results
        testResult, testLabel = evaluation(testloader, net)

    elif CNNName == 'resnet':
        X_train_img = augmentedDataset[0].transpose(0, 2, 3, 1)
        X_test_img = augmentedDataset[1].transpose(0, 2, 3, 1)
        y_train = augmentedDataset[2]
        y_test = augmentedDataset[3]

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        num_classes = np.unique(y_train_enc).size

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        net = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False, verbose=False)
        net.fc = nn.Linear(2048, num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = net.to(device)

        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

        X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float().to(device)
        y_train_tensor = torch.from_numpy(le.fit_transform(y_train)).to(device)

        X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float().to(device)
        y_test_tensor = torch.from_numpy(le.transform(y_test)).to(device)

        batch_size = 10

        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

        # 50 epoch batch_size=10 variance=1
        for epoch in range(50):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            # print epoch statistics
            # print('[%d] loss: %.3f' %
            #       (epoch + 1, running_loss / len(X_train_tensor) * batch_size))


        testset = TensorDataset(X_test_tensor, y_test_tensor)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
        # Run evaluation function to get results
        testResult, testLabel = evaluation(testloader, net)


    elif CNNName == 'densenet':
        X_train_img = augmentedDataset[0].transpose(0, 2, 3, 1)
        X_test_img = augmentedDataset[1].transpose(0, 2, 3, 1)
        y_train = augmentedDataset[2]
        y_test = augmentedDataset[3]

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        num_classes = np.unique(y_train_enc).size

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        net = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=False, verbose=False)

        net.fc = nn.Linear(18085, num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = net.to(device)

        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

        X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float().to(device)
        y_train_tensor = torch.from_numpy(le.fit_transform(y_train)).to(device)

        X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float().to(device)
        y_test_tensor = torch.from_numpy(le.transform(y_test)).to(device)

        X_train_tensor.dtype

        batch_size = 10

        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

        # 50 epoch batch_size=10 variance=0 annotation=yes
        for epoch in range(50):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            # print epoch statistics
            # print('[%d] loss: %.3f' %
            #       (epoch + 1, running_loss / len(X_train_tensor) * batch_size))

        testset = TensorDataset(X_test_tensor, y_test_tensor)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        # Run evaluation function to get results
        testResult, testLabel = evaluation(testloader, net)


    else:
        print("Please enter a correct CNN method name.")

    # Return all results
    return [testResult, testLabel]

def evaluation(loader, net):
    net.eval()
    with torch.no_grad():
        result = []
        true_labels = []
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            # forward + backward + optimize
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs, 1)
            result.append(train_predicted.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
    result1 = [b for a in result for b in a]
    true_label1 = [b for a in true_labels for b in a]
    return result1, true_label1

# # test CNNTrain
# x = CNNTrain(1, "vgg16")
# x = np.array(x)
# np.save('test_Ariel.npy', x)

