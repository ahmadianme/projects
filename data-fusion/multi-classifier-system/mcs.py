import math
import time
from pprint import pprint

import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import to_categorical




# featuresFrom = 'file'
featuresFrom = 'extract'




if featuresFrom == 'extract':
    dataFrame = pandas.read_csv('SisFall_dataset.csv')




    saplingRate = 200
    sampleCount = len(dataFrame)
    start = 0


    newData = {
        'f1': [],
        'f2': [],
        'f3': [],
        'label': [],
    }

    print(int(sampleCount/saplingRate))

    while sampleCount >= start + saplingRate:
        newData['label'].append(dataFrame[['Situation']][start:start+saplingRate].mode().to_numpy()[0][0])

        data = dataFrame[['ITG3200_x', 'ITG3200_y', 'ITG3200_z']][start:start+saplingRate].to_numpy()

        newData['f1'].append(np.mean(np.sqrt(np.sum(data**2, axis=1))))
        newData['f2'].append(np.mean(np.arctan2(np.sqrt(np.sum(np.square(data[:, :2]), axis=1)), -data[:, 2:])))
        newData['f3'].append(np.mean(np.sqrt(np.var(data, axis=0))))

        start += int(saplingRate / 2)

    newDataFrame = pandas.DataFrame.from_dict(newData)

    newDataFrame.to_csv('features.csv')

    dataFrame = newDataFrame
else:
    dataFrame = pandas.read_csv('features.csv', usecols = ['f1','f2', 'f3', 'label'])









dataX = dataFrame.loc[:, dataFrame.columns != 'label']
dataY = dataFrame['label'].replace(['Fall', 'Not Fall'], [1, 0])

trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.2, random_state=0, shuffle=True)



baseClassifiers = []

baseClassifiers.append(RandomForestClassifier(n_estimators=5)) # 94%
baseClassifiers.append(KNeighborsClassifier(n_neighbors=50)) # 95%
baseClassifiers.append(XGBClassifier(n_estimators=20, max_depth=5, learning_rate=1, objective='binary:logistic')) # 93%



classifiers = baseClassifiers.copy()
predictions = []
precisions = []



for i, classifier in enumerate(classifiers):
    startTime = time.time()

    classifier.fit(trainX, trainY)

    predictions.append(classifier.predict(testX))

    precisions.append(precision_score(testY, predictions[i]))

    print('Training and testing classifier '+ str(i+1) +'. Time: '+ str(time.time() - startTime))



print()
print('Classifiers precision:')
pprint(precisions)
print()
print()
print()
print()






baggingCopyCount = 1

baggingClassifiers = []
baggingPredictions = []
baggingPrecisions = []

for i in range(baggingCopyCount):
    baggingClassifiers += baseClassifiers.copy()



for i, baggingClassifier in enumerate(baggingClassifiers):
    startTime = time.time()

    indices = np.random.choice(range(len(trainX)), size=len(trainX), replace=True)
    baggingTrainX = trainX.loc[trainX.index[indices]]
    baggingTrainY = trainY.loc[trainY.index[indices]]

    baggingClassifier.fit(baggingTrainX, baggingTrainY)

    baggingPredictions.append(baggingClassifier.predict(testX))

    baggingPrecisions.append(precision_score(testY, baggingPredictions[i]))

    print('Training and testing baggingClassifier '+ str(i+1) +'. Time: '+ str(time.time() - startTime))

print()
print('baggingClassifiers precision:')
pprint(baggingPrecisions)
print()
print()
print()
print()
print()







weights = []

for precision in baggingPrecisions:
    weights.append(np.log(precision / (1 - precision)))

weights /= np.sum(weights)






mvPredictions = []
mvPrecision = []
wmvPredictions = []
wmvPrecision = []


for prediction in np.transpose(baggingPredictions):
    if np.count_nonzero(prediction) > math.floor(len(prediction) / 2) + 1:
        mvPredictions.append(1)
    else:
        mvPredictions.append(0)

    fallIndicies = np.where(prediction == 1)
    fallScore = np.sum(np.array(weights)[fallIndicies])

    notFallIndicies = np.where(prediction == 0)
    notFallScore = np.sum(np.array(weights)[notFallIndicies])

    if fallScore >= notFallScore:
        wmvPredictions.append(1)
    else:
        wmvPredictions.append(0)



mvPrecision = precision_score(testY.to_numpy(), np.array(mvPredictions))
wmvPrecision = precision_score(testY.to_numpy(), np.array(wmvPredictions))

print()
print('Majority Voting Precision: ' + str(wmvPrecision))
print('Weighted Majority Voting Precision: ' + str(mvPrecision))
print()
print()
print()
print()
print()
















initialModels = [(str(i), clf) for i, clf in enumerate(baseClassifiers)]

metaModel = LogisticRegression()
stack_model = StackingClassifier(estimators=initialModels, final_estimator=metaModel)
stackingPrecision = cross_val_score(stack_model, trainX, trainY, cv=5, scoring='precision')

print()
print('Stacking Precision: ' + str(stackingPrecision))
print()
print()
print()
print()
print()








# start = 0
# saplingRate = 100
# sampleCount = len(dataX)

# cnnDataX = []
# cnnDataY = []
# # cnnData = []
#
# while sampleCount >= start + saplingRate:
#     # cnnDataX.append(dataFrame[['f1']][start:start+saplingRate].to_numpy().flatten().tolist())
#     cnnDataX.append(dataFrame[['f1', 'f2', 'f3']][start:start+saplingRate].to_numpy().tolist())
#     cnnDataY.append(int(dataFrame[['label']][start:start+saplingRate].mode().replace(['Fall', 'Not Fall'], [1, 0]).to_numpy()[0][0]))
#
#     # data = dataFrame[['f1', 'f2', 'f3']][start:start+saplingRate].to_numpy().tolist()
#     #
#     # print(data)
#     # exit()
#     # data = np.append(data, int(dataFrame[['label']][start:start+saplingRate].mode().replace(['Fall', 'Not Fall'], [1, 0]).to_numpy()[0][0]))
#     # cnnData.append(data)
#     # print(cnnDataX)
#     # print(cnnDataY)
#     # exit()
#
#     start += saplingRate
#
#
# # print(np.array(cnnDataX).shape)
# # print(np.array(cnnDataY).shape)
# # print(np.array(cnnData).shape)
#
# cnnTrainX, cnnTestX, cnnTrainY, cnnTestY = train_test_split(cnnDataX, cnnDataY, test_size=0.2, random_state=0, shuffle=True)



# print(np.array(cnnDataX).shape)
# print(np.array(cnnDataY).shape)
#
#
# print(np.array(cnnTrainX).shape)
# # print(np.array(cnnTestX).shape)
# print(np.array(cnnTrainY).shape)
# # print(np.array(cnnTestY).shape)
# # print(np.array(cnnTestY))

trainY = to_categorical(trainY)
# testY = to_categorical(testY)

trainX = np.array(trainX).tolist()
trainY = np.array(trainY).tolist()

testX = np.array(testX).tolist()
# testY = np.array(testY).tolist()

print(np.array(trainX).shape)
print(np.array(trainY).shape)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(Dropout(0.5))
# model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=1)

# _, cnnAccuracy = model.evaluate(testX, np.array(to_categorical(testY)).tolist(), batch_size=32, verbose=1)
# print('CNN Accuracy: ' + str(cnnAccuracy))

cnnPredictions = model.predict(testX)
cnnPredictions = np.argmax(cnnPredictions, axis=1)

cnnPrecision = precision_score(testY.to_numpy(), np.array(cnnPredictions))

print()
print('CNN Precision: ' + str(cnnPrecision))
print()
print()
print()
print()
print()








baggingPredictions.append(cnnPredictions)
baggingPrecisions.append(cnnPrecision)



weights = []

for precision in baggingPrecisions:
    weights.append(np.log(precision / (1 - precision)))

weights /= np.sum(weights)






mvPredictions = []
mvPrecision = []
wmvPredictions = []
wmvPrecision = []


for prediction in np.transpose(baggingPredictions):
    if np.count_nonzero(prediction) > math.ceil(len(prediction) / 2) + 1:
        mvPredictions.append(1)
    else:
        mvPredictions.append(0)

    fallIndicies = np.where(prediction == 1)
    fallScore = np.sum(np.array(weights)[fallIndicies])

    notFallIndicies = np.where(prediction == 0)
    notFallScore = np.sum(np.array(weights)[notFallIndicies])

    if fallScore > notFallScore:
        wmvPredictions.append(1)
    else:
        wmvPredictions.append(0)



mvPrecision = precision_score(testY.to_numpy(), np.array(mvPredictions))
wmvPrecision = precision_score(testY.to_numpy(), np.array(wmvPredictions))


print()
print('CNN Majority Voting Precision: ' + str(wmvPrecision))
print('CNN Weighted Majority Voting Precision: ' + str(mvPrecision))
print()
print()
print()
print()
print()
