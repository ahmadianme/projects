import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import itertools
from itertools import cycle

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA






dataDir = 'data/'

task = 'gender'
# task = 'emotion'

featureCombination = 1

applyNormalization = True
applyStandardization = True
applyPCA = False
applyLDA = False

pcaComponentCount = 50

# classifier = clf = svm.SVC(kernel='linear')
# classifier = clf = svm.SVC(gamma=2, C=1)
classifier = clf = svm.SVC(kernel='rbf')
# classifier = clf = GaussianNB()
# classifier = clf = GaussianProcessClassifier(1.0 * RBF(1.0))
# classifier = clf = QuadraticDiscriminantAnalysis()
# classifier = MLPClassifier(activation='relu', hidden_layer_sizes=500, batch_size=200, learning_rate='invscaling', solver='adam')
# classifier = tree.DecisionTreeClassifier()
# classifier = RandomForestClassifier(n_estimators=1500)
# classifier = AdaBoostClassifier(n_estimators=1000)
# classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)










preprocessors = {}



def normalizeData(data, preprocessors):
    if 'minMax' not in preprocessors:
        preprocessors['minMax'] = MinMaxScaler()

    data = preprocessors['minMax'].fit_transform(data.values)

    return pd.DataFrame(data), preprocessors



def unnormalizeData(data, preprocessors):
    data = preprocessors['minMax'].inverse_transform(data)

    return pd.DataFrame(data)



def scaleData(data, preprocessors):
    if 'zscore' not in preprocessors:
        preprocessors['zscore'] = StandardScaler()

    data = preprocessors['zscore'].fit_transform(data.values)

    return pd.DataFrame(data), preprocessors



def unscaleData(data, preprocessors):
    data = preprocessors['zscore'].inverse_transform(data)

    return pd.DataFrame(data)



def doPCA(data, preprocessors, componentCount):
    if 'pca' not in preprocessors:
        preprocessors['pca'] = PCA(n_components=componentCount)

    data = preprocessors['pca'].fit_transform(data.values)

    return pd.DataFrame(data), preprocessors



def doLDA(data, preprocessors, dataY=None):
    if 'lda' not in preprocessors:
        preprocessors['lda'] = LDA()

    if dataY is not None:
        data = preprocessors['lda'].fit_transform(data.values, dataY)
    else:
        data = preprocessors['lda'].transform(data.values)

    return pd.DataFrame(data), preprocessors



def plotConfusionMatrix(matrix, classes, normalize=False):
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tickMarks = np.arange(len(classes))
    plt.xticks(tickMarks, classes, rotation=90)
    plt.yticks(tickMarks, classes)

    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j],
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()











print('Loading data files...\n')

data = pd.read_csv(dataDir + 'features-final.csv')
# data = pd.read_csv(dataDir + 'features_downsample.csv')




featureZCR = data.iloc[:, 0:1]
featureChromaShift = data.iloc[:, 1:13]
featureMFCC = data.iloc[:, 13:33]
featureRMS = data.iloc[:, 33:34]
featureMel = data.iloc[:, 34:162]
featureAmplitudEnvelope = data.iloc[:, 162:163]
featureSpectralCentroid = data.iloc[:, 163:164]
featureSpectralBandwidth = data.iloc[:, 164:165]

featureMFCC = featureMFCC.join(featureMel)

# dataX = data.iloc[:, :-1]

if featureCombination == 1:
    dataX = featureMFCC
    dataX = dataX.join(featureZCR)
    dataX = dataX.join(featureChromaShift)
    dataX = dataX.join(featureRMS)
    dataX = dataX.join(featureSpectralCentroid)
    dataX = dataX.join(featureSpectralBandwidth)
else:
    dataX = featureMFCC
    dataX = dataX.join(featureZCR)
    dataX = dataX.join(featureRMS)
    dataX = dataX.join(featureChromaShift)
    dataX = dataX.join(featureAmplitudEnvelope)




if task == 'emotion':
    dataY = data.iloc[:, -1]
    dataY.replace({4: 3, 3: 2, 2: 1, 1: 0}, inplace=True)
else:
    dataGender = pd.read_csv(dataDir + 'genders.csv')
    dataY = dataGender.iloc[:, -1]
    dataY.replace({'m': 0, 'f': 1}, inplace=True)

if task == 'emotion':
    id2label = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'neutral'}
    label2id = {'angry': 0, 'happy': 1, 'sad': 2, 'neutral': 3}
else:
    id2label = {0: 'male', 1: 'female'}
    label2id = {'male': 0, 'female': 1}

allLabels = list(set(dataY.values))

allLabelsTitles = []
for label in id2label:
    allLabelsTitles.append(id2label[label])



trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.3, random_state=0, shuffle=True)


if applyNormalization == True:
    print('Normilizing and scaling data...\n')
    trainX, preprocessors = normalizeData(trainX, preprocessors)
    testX, preprocessors = normalizeData(testX, preprocessors)

if applyStandardization == True:
    print('Scaling and Standardizing...\n')
    trainX, preprocessors = scaleData(trainX, preprocessors)
    testX, preprocessors = scaleData(testX, preprocessors)

if applyPCA == True:
    print('Reducing features using PCA...\n')
    trainX, preprocessors = doPCA(trainX, preprocessors, pcaComponentCount)
    testX, preprocessors = doPCA(testX, preprocessors, pcaComponentCount)

if applyLDA == True:
    print('Reducing features using LDA...\n')
    trainX, preprocessors = doLDA(trainX, preprocessors, trainY)
    testX, preprocessors = doLDA(testX, preprocessors)







print('Training the model...\n')
startTime = time.time()

print(classifier.fit(trainX, trainY).score(trainX, trainY))

print('Training done. Time: ' + str(time.time() - startTime) + '\n')

print('Testing the model...\n')
startTime = time.time()

predictions = classifier.predict(testX)
print('Testing done. Time: ' + str(time.time() - startTime) + '\n')

# Classification metrics

classification_report = classification_report(testY, predictions)

print('Classification Report\n')
print(classification_report)
print()










confusionMatrix = confusion_matrix(testY, predictions, labels=allLabels)
plotConfusionMatrix(confusionMatrix, allLabelsTitles)










fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(allLabels)):
    testYOVA = np.asarray(testY.values.copy())
    predictionsOVA = np.asarray(predictions.copy())

    testYOVA[testYOVA == i] = 10
    testYOVA[testYOVA != 10] = 0
    testYOVA[testYOVA == 10] = 1

    predictionsOVA[predictionsOVA == i] = 10
    predictionsOVA[predictionsOVA != 10] = 0
    predictionsOVA[predictionsOVA == 10] = 1

    fpr[i], tpr[i], _ = roc_curve(testYOVA, predictionsOVA)
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(allLabels))]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(len(allLabels)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= len(allLabels)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(6, 6))


plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (Area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["green", "darkorange", "red", "navy"])

for i, color in zip(range(len(allLabels)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label="ROC curve of class {0} (Area = {1:0.2f})".format(id2label[i], roc_auc[i]),)

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
