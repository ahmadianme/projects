import torch
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
from numpy import vstack
from pandas import read_csv
from sklearn.metrics import accuracy_score
from torch.utils.data import  DataLoader
from torch.nn import Linear, ReLU, LogSoftmax, Sigmoid, Tanh, Module, CrossEntropyLoss, MSELoss, Softmax, functional
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm







class Network(Module):
    def __init__(self):
        super(Network, self).__init__()

        self.hidden1 = Linear(28*28, 512)
        self.activation1 = hiddenLayerActivation()

        self.hidden2 = Linear(512, 256)
        self.activation2 = hiddenLayerActivation()

        self.outout = Linear(256, 10)
        self.outputActivation = outputLayerActivation()


    def forward(self, X):
        X = self.hidden1(X)
        X = self.activation1(X)

        X = self.hidden2(X)
        X = self.activation2(X)

        X = self.outout(X)
        X = functional.log_softmax(X, dim=1)

        return X







def train(trainDataLoader, validationDataLoader:None, model):
    criterion = lossFunction()

    trainLosses = []
    validationLosses = []
    trainAccuracies = []
    validationAccuracies = []

    iter_wrapper = (lambda x: tqdm(x, total=len(trainDataLoader)))

    for epoch in range(epochs):
        batchTotalLoss = 0
        predictions, actuals = list(), list()
        for i, (inputs, targets) in iter_wrapper(enumerate(trainDataLoader)):
            model.train()

            optimizer.zero_grad()

            yhat = model(inputs)

            yhatArgmax = torch.zeros(batchSize)
            for i, y in enumerate(yhat):
                yhatArgmax[i] = y.argmax()


            loss = criterion(yhat, targets)

            loss.backward()
            optimizer.step()

            batchTotalLoss += loss.item()

            yhat = yhat.detach().numpy()
            targets = targets.numpy()

            for cnt in range(len(targets)):
                predictions.append(yhat[cnt].argmax())
                if onehotLabels:
                    actuals.append(targets[cnt].argmax())
                else:
                    actuals.append(targets[cnt])


        print('Epoch ' + str(epoch + 1) + ' Loss: ' + str(batchTotalLoss / len(trainDataLoader)))

        predictions, actuals = vstack(predictions), vstack(actuals)
        trainAccuracies.append(accuracy_score(actuals, predictions))


        trainLosses.append(batchTotalLoss / len(trainDataLoader))

        validationLosses, validationAccuracies, confusionMatrix = None, None, None

        if validationDataLoader != None:
            accuracyTmp, lossTmp, confusionMatrix = evaluate(validationDataLoader, model)

            validationAccuracies.append(accuracyTmp)
            validationLosses.append(lossTmp.item())


    return trainLosses, trainAccuracies, validationLosses, validationAccuracies, confusionMatrix





def evaluate(dataLoader, model):
    predictions, actuals = list(), list()
    totalLoss = 0
    confusionArray = {}
    samplesCount = {}
    confusionMatrix = torch.zeros(10, 10)

    for i, (inputs, targets) in enumerate(dataLoader):
        model.eval()

        yhat = model(inputs)

        criterion = lossFunction()
        loss = criterion(yhat, targets)
        totalLoss += loss.item()

        yhat = yhat.detach().numpy()

        targets = targets.numpy()

        for cnt in range(len(targets)):
            predictions.append(yhat[cnt].argmax())
            if onehotLabels:
                actuals.append(targets[cnt].argmax())
            else:
                actuals.append(targets[cnt])

    for cnt in range(len(actuals)):
        confusionMatrix[predictions[cnt], actuals[cnt]] += 1

    predictions, actuals = vstack(predictions), vstack(actuals)
    acc = accuracy_score(actuals, predictions)

    totalLoss /= len(dataLoader)

    return acc, totalLoss, confusionMatrix












# hyper parameters ########################################################################################
batchSize = 256
epochs = 20
onehotLabels = True
flattenImages = True
lossFunction = MSELoss
hiddenLayerActivation = Sigmoid
outputLayerActivation = LogSoftmax
model = Network()
optimizer = Adam(model.parameters(), lr=0.01)
###########################################################################################################






def oneHot(x):
    label = torch.zeros(10)
    label[x] = 1
    return label

# normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]])
flatten = transforms.Lambda(lambda x: torch.flatten(x))
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), flatten])
if flattenImages:
    transform = transforms.Compose([transforms.ToTensor(), flatten])
else:
    transform = transforms.Compose([transforms.ToTensor()])

onehot = transforms.Lambda(lambda x:oneHot(x))
transformTarget = transforms.Compose([onehot])

trainData = torchvision.datasets.FashionMNIST(
    root='datasets/FashionMNIST',
    train=True,
    download=True,
    transform=transform,
    target_transform=transformTarget
)

testData = torchvision.datasets.FashionMNIST(
    root='datasets/FashionMNIST',
    train=False,
    download=True,
    transform=transform,
    target_transform=transformTarget
)

trainDataLoader = DataLoader(trainData,batch_size=batchSize)
testDataLoader = DataLoader(testData,batch_size=batchSize)
images, labels = next(iter(trainDataLoader))


startTime = time.time()
trainLosses, trainAccuracies, _, _, _= train(trainDataLoader, None, model)
endTime = time.time()
trainingTime = endTime - startTime
print('Training Time: ' + str(trainingTime))
print()
print('Train Loss: ' + str(trainLosses))
print('Train Accuracy: ' + str(trainAccuracies))
print()

testAccuracy, testLoss, testConfusionMatrix = evaluate(testDataLoader, model)

# print('Validation Confusion Matrix')
# print(validationConfusionMatrix)
# print()
print('Test Confusion Matrix')
print(testConfusionMatrix)
print()
print('Test Loss: ' + str(testLoss))
print('Test Accuracy: ' + str(testAccuracy))



plt.plot(range(len(trainLosses)), trainLosses, label = "Train Loss")
# plt.plot(range(len(validationLosses)), validationLosses, label = "Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.legend()
plt.show()


plt.plot(range(len(trainAccuracies)), trainAccuracies, label = "Train Accuracy")
# plt.plot(range(len(validationAccuracies)), validationAccuracies, label = "Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Plot')
plt.legend()
plt.show()


# ax = sns.heatmap( validationConfusionMatrix , linewidth = 0.5 , cmap = 'rocket_r' )
# plt.title( "Validation Confusion Matrix")
# plt.show()


ax = sns.heatmap( testConfusionMatrix , linewidth = 0.5 , cmap = 'rocket_r' )
plt.title( "Test Confusion Matrix")
plt.show()
