import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier




dataset = pd.read_csv('cancer.csv', index_col='Sample Code Number', na_values='?')
dataset.dropna(inplace=True)
dataset = dataset.to_numpy(dtype='float16')

x = dataset[:, :-1]
y = dataset[:, -1]



logisticRegressionCLF = LogisticRegression(random_state=0)
desicionTreeCLF = DecisionTreeClassifier(random_state=0)
svmCLF = SVC(kernel='rbf')



kFold = KFold(n_splits=10, shuffle=True, random_state=53)
accuracies = []

for trainIndicies, testIndicies in kFold.split(x):
    trainX = x[trainIndicies,:]
    testX = x[testIndicies,:]

    trainY = y[trainIndicies]
    testY = y[testIndicies]

    clf = VotingClassifier(estimators=[
        ('LogisticRegression', logisticRegressionCLF),
        ('DecisionTreeClassifier', desicionTreeCLF),
        ('SVM', svmCLF)],
        voting='hard')

    clf.fit(trainX, trainY)
    testPredictions = clf.predict(testX)
    accuracies.append(accuracy_score(testY, testPredictions))

print(accuracies)
print('Average Accuracy: %f' % np.mean(accuracies))
