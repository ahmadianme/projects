import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import scipy.io as sio
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

import pyeeg as pe

dataDir = 'data/'
dataFaceFileName = dataDir + 'S2T2B1.mat'
dataPianoFileName = dataDir + 'S2T2B2.mat'

channel = range(0, 126)
band = [4, 8, 12, 16, 25, 45]
window_size = 250
step_size = 50
sample_rate = 500

extractFeatures = True





dataY = np.concatenate((np.zeros(45), np.ones(45)))






def featureExtract (dataFileName, channel, band, window_size, step_size, sample_rate):
    allMetas = []
    meta = []
    allData = sio.loadmat(dataFileName)
    allData = allData['a']

    for i in range (0, 45):
        data = allData[i]
        start = 0;


        meta_array = []

        while start + window_size < data.shape[1]:
            # print(start)

            meta_data = []
            for j in channel:
                X = data[j][start : start + window_size]
                Y = pe.bin_power(X, band, sample_rate)
                meta_data = meta_data + list(Y[0])

            meta_array.append(np.array(meta_data))

            start = start + step_size
        meta.append(np.array(meta_array))


    meta = np.array(meta)

    return meta




if extractFeatures == True:
    print('Loading data files and extracting features...')

    featuresFace = featureExtract(dataFaceFileName, channel, band, window_size, step_size, sample_rate)
    # print(featuresFace.shape)

    featuresPiano = featureExtract(dataPianoFileName, channel, band, window_size, step_size, sample_rate)
    # print(featuresPiano.shape)

    dataX = np.concatenate((featuresFace, featuresPiano), axis=0)

    with open('data/dataX.bin', 'wb') as handle:
        pickle.dump(dataX, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Features has been saved to dataX.bin')
else:
    print('Loading features from dataX.bin')

    with open('data/dataX.bin', 'rb') as handle:
        dataX = pickle.load(handle)




print('Training the model...\n')
startTime = time.time()

trainX = dataX
trainY = dataY



numSamples = int((3500/window_size*window_size/500-window_size/500)*10)
startSecond = 0.25
endSecond = 6.75



seedCount = 30

accuraciesOverTime = np.zeros([numSamples, seedCount])

for seed in range(seedCount):
    seedInt = random.randint(0, 1000)

    print('Training using seed: ' + str(seedInt))

    accuracyOverTime = []

    for i in range(numSamples):
        subTrainX = []
        for trainXItem in trainX:
            subTrainX.append(trainXItem[i])

        scaler = MinMaxScaler()

        subTrainX = scaler.fit_transform(subTrainX)

        classifier = LogisticRegression(random_state=seedInt)

        scores = cross_val_score(classifier, subTrainX, dataY, cv=5, scoring='accuracy')
        accuracyOverTime.append(np.mean(np.array(scores)))

    for j in range(numSamples):
        accuraciesOverTime[j][seed] = accuracyOverTime[j]

print('Training and testing models done. Time: ' + str(time.time() - startTime) + '\n')

accuracyMean = np.mean(accuraciesOverTime, axis=1)

print('Best Accuracy: ' + str(np.max(accuracyMean)))
print('Best Accuracy in: ' + str((np.argmax(accuracyMean) + startSecond*10) / 10))




ci = 1.96 * np.std(accuracyMean)/np.sqrt(len(accuracyMean))




plt.plot([x / 100.0 for x in range(int(startSecond*100), int(endSecond*100), 10)], accuracyMean, lw=2)
plt.fill_between([x / 100.0 for x in range(int(startSecond*100), int(endSecond*100), 10)], (accuracyMean-ci), (accuracyMean+ci), color='b', alpha=.25)
plt.xlabel("Time (Seconds)")
plt.ylabel("Accuracy")
plt.title("Accuracy over Time Plot")
plt.show()
