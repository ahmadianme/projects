import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler




def drawPlot(x1Data, y1Data, label1, x2Data, y2Data, label2, title, xLabel=None, yLabel=None):

    plt.title(title)
    plt.scatter(x1Data, y1Data, label=label1)
    plt.scatter(x2Data, y2Data, label=label2)
    plt.legend()
    plt.show()



def drawPlot2(x1Data, y1Data, label1, x2Data, y2Data, label2, contourX, contourY, contourProbabilities, title, xLabel=None, yLabel=None):
    plt.xlabel(xLabel if xLabel else "x")
    plt.ylabel(yLabel if yLabel else "y")
    plt.title(title)
    plt.scatter(x1Data, y1Data, label=label1)
    plt.scatter(x2Data, y2Data, label=label2)
    plt.contour(contourX, contourY, contourProbabilities, levels=[.5], vmin=0.0, vmax=.6)
    plt.legend()
    plt.show()



def drawPlot3(x1Data, y1Data, label1, x2Data, y2Data, label2, title, xLabel=None, yLabel=None):
    plt.xlabel(xLabel if xLabel else "x")
    plt.ylabel(yLabel if yLabel else "y")
    plt.title(title)
    plt.plot(x1Data, y1Data, label=label1)
    plt.plot(x2Data, y2Data, label=label2)
    plt.legend()
    plt.show()



def generateFeatures(X, degree):
    S = []

    for i in range(1, degree):
        for j in itertools.combinations_with_replacement(range(2), i):
            S.append(j)

    generated = np.zeros((len(X), len(S)))

    for i, s in enumerate(S):
        generated[:, i] = (X[:, 0] ** np.sum(np.array(s) == 0)) + (X[:, 1] ** np.sum(np.array(s) == 1))

    return generated




def generateAndTest():
    drawPlot(x[y == 0, 0], x[y == 0, 1], 'Class 1', x[y == 1, 0], x[y == 1, 1], 'Class 2', 'Generated Data')

    indexShuffled = np.random.permutation(len(x))
    splitSize = int(len(x) * .75)
    trainShuffledIndex = indexShuffled[splitSize:]
    testShuffledIndex = indexShuffled[:splitSize]
    xTrain, yTrain = x[trainShuffledIndex], y[trainShuffledIndex].flatten()
    xTest, yTest = x[testShuffledIndex], y[testShuffledIndex].flatten()



    testAccuracy = []
    trainAccuracy = []

    for degree in range(1, 16):
        xTrainFeatures = generateFeatures(xTrain, degree + 1)
        xTestFeatures = generateFeatures(xTest, degree + 1)

        xMax = np.max(xTrainFeatures, axis=0)
        xMin = np.min(xTrainFeatures, axis=0)
        xTrainFeaturesNormalized = (xTrainFeatures - xMin) / (xMax - xMin)
        xTestFeaturesNormalized = (xTestFeatures - xMin) / (xMax - xMin)

        xTrainFeaturesNormalized = np.c_[np.ones((len(xTrain), 1)), xTrainFeaturesNormalized]
        xTestFeaturesNormalized = np.c_[np.ones((len(xTest), 1)), xTestFeaturesNormalized]

        parameter = np.random.randn(xTrainFeaturesNormalized.shape[1], 1)
        for _ in range(250):
            yProbabilities = 1.0 / (1.0 + np.exp(-1 * (xTrainFeaturesNormalized @ parameter)))
            error = yProbabilities - yTrain[:, np.newaxis]
            gradient = xTrainFeaturesNormalized.T @ error + parameter
            parameter -= 0.5 * (1/len(xTrainFeaturesNormalized))*gradient

        yTrainProbabilities = 1.0 / (1.0 + np.exp(-1 * (xTrainFeaturesNormalized @ parameter)))
        yTrainPredictions = (yTrainProbabilities>0.5).flatten().astype(int)
        trainAccuracy.append(np.sum(yTrainPredictions==yTrain)/len(xTrain))

        yTestProbabilities = 1.0 / (1.0 + np.exp(-1 * (xTestFeaturesNormalized @ parameter)))
        yTestPredictions = (yTestProbabilities>0.5).flatten().astype(int)
        testAccuracy.append(np.sum(yTestPredictions==yTest)/len(xTest))

    drawPlot3(range(1, 16), trainAccuracy, 'Train', range(1, 16), testAccuracy, 'Test', 'Train Accuracy - Test Accuracy', 'Degree', 'Accuracy')



    xTrainFeatures = generateFeatures(xTrain, 16)
    xTestFeatures = generateFeatures(xTest, 16)

    xMax = np.max(xTrainFeatures, axis=0)
    xMin = np.min(xTrainFeatures, axis=0)
    xTrainFeaturesNormalized = (xTrainFeatures - xMin) / (xMax - xMin)
    xTestFeaturesNormalized = (xTestFeatures - xMin) / (xMax - xMin)

    xTrainFeaturesNormalized = np.c_[np.ones((len(xTrain), 1)), xTrainFeaturesNormalized]
    xTestFeaturesNormalized = np.c_[np.ones((len(xTest), 1)), xTestFeaturesNormalized]

    parameter = np.random.randn(xTrainFeaturesNormalized.shape[1], 1)
    for _ in range(250):
        yProbabilities = 1.0 / (1.0 + np.exp(-1 * (xTrainFeaturesNormalized @ parameter)))
        error = yProbabilities - yTrain[:, np.newaxis]
        gradient = xTrainFeaturesNormalized.T @ error + parameter
        parameter -= 0.5 * (1 / len(xTrainFeaturesNormalized)) * gradient



    gridX, gridY = np.mgrid[np.min(x1):np.max(x1):.1, np.min(x2):np.max(x2):.1]
    grid = np.c_[gridX.ravel(), gridY.ravel()]
    gridFeatures = generateFeatures(grid, degree + 1)
    scaler = MinMaxScaler()
    scaler.fit_transform(xTrainFeatures)
    gridNormalized = scaler.transform(gridFeatures)
    gridNormalized = np.c_[np.ones((len(gridNormalized), 1)), gridNormalized]
    probabilities = (1.0 / (1.0 + np.exp(-1 * (gridNormalized @ parameter)))).reshape(gridX.shape)

    drawPlot2(x1[:, 0], x1[:, 1], 'class 1', x2[:, 0], x2[:, 1], 'class 2', gridX, gridY, probabilities, 'Generated Data Classifier')







y = np.concatenate([np.zeros(200), np.ones(200)])
x1 = []

for i in range(200):
    r = np.random.uniform(4, 9)
    a = np.random.uniform(0, 2 * np.pi)
    x1.append((1.5 + r * np.cos(a), 0.0 + r * np.sin(a)))

x2 = []

for i in range(200):
    r = np.random.uniform(0, 6)
    a = np.random.uniform(0, 2 * np.pi)
    x2.append((1.5 + r * np.cos(a), 0.0 + r * np.sin(a)))

x = np.vstack([np.array(x1), np.array(x2)])

x1 = np.array(x1)
x2 = np.array(x2)

generateAndTest()






y = np.concatenate([np.zeros(100), np.ones(200)])
x2 = []

for i in range(200):
    r = np.random.uniform(2, 6)
    a = np.random.uniform(0, 2 * np.pi)
    x2.append((1.5 + r * np.cos(a), 0.0 + r * np.sin(a)))

x1 = np.array(list(zip(np.random.normal(1, 1, 100), np.random.normal(0, 1, 100))))
x2 = np.array(x2)
x = np.vstack([x1, x2])

random = np.arange(len(x))
x = x[random]
y = y[random]

generateAndTest()
