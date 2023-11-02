import pandas as pd
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt







dataset = pd.read_csv('dataset.csv')

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.3)











trainResults = {}
testResults = {}

for C in [1, 100, 1000]:
    clf = SVC(C=C, kernel='rbf')
    clf.fit(trainX, trainY)

    trainPredictions = clf.predict(trainX)
    testPredictions = clf.predict(testX)
    trainResults[C] = accuracy_score(y_pred=trainPredictions, y_true=trainY)
    testResults[C] = accuracy_score(y_pred=testPredictions, y_true=testY)

print('Examining RBF Kernel:')
print('Train Results:')
print(trainResults)
print('Test Results:')
print(testResults)
print()





trainResults = {}
testResults = {}

for C in [1, 100, 1000]:
    clf = SVC(C=C, kernel='linear')
    clf.fit(trainX, trainY)

    trainPredictions = clf.predict(trainX)
    testPredictions = clf.predict(testX)

    trainResults[C] = accuracy_score(y_pred=trainPredictions, y_true=trainY)
    testResults[C] = accuracy_score(y_pred=testPredictions, y_true=testY)

print('Examining Linear Kernel:')
print('Train Results:')
print(trainResults)
print('Test Results:')
print(testResults)
print()





trainResults = {}
testResults = {}

for C in [1, 100]:
    clf = SVC(C=C, kernel='poly')
    clf.fit(trainX, trainY)

    trainPredictions = clf.predict(trainX)
    testPredictions = clf.predict(testX)

    trainResults[C] = accuracy_score(y_pred=trainPredictions, y_true=trainY)
    testResults[C] = accuracy_score(y_pred=testPredictions, y_true=testY)

print('Examining Polynomial Kernel:')
print('Train Results:')
print(trainResults)
print('Test Results:')
print(testResults)
print()





trainResults = {}
testResults = {}

for C in [1, 100]:
    clf = SVC(C=C, kernel='sigmoid')
    clf.fit(trainX, trainY)

    trainPredictions = clf.predict(trainX)
    testPredictions = clf.predict(testX)

    trainResults[C] = accuracy_score(y_pred=trainPredictions, y_true=trainY)
    testResults[C] = accuracy_score(y_pred=testPredictions, y_true=testY)

print('Examining Sigmoid Kernel:')
print('Train Results:')
print(trainResults)
print('Test Results:')
print(testResults)
print()














print()
print()
print('Grid Search for finding best parameters:')


print()
print('RBF kernel:')

clf = GridSearchCV(SVC(kernel='rbf'), {'C': [1, 10, 100, 500], 'gamma': [0.1, 0.3, 0.5, 0.7, 0.9]})
clf.fit(x, y)
results = pd.DataFrame(clf.cv_results_).loc[:, ['param_C', 'param_gamma', 'mean_test_score']]
print(results.sort_values(by='mean_test_score', ascending=False))



print()
print('Linear kernel:')

clf = GridSearchCV(SVC(kernel='linear'), {'C': [1, 10, 100, 1000]})
clf.fit(x, y)
results = pd.DataFrame(clf.cv_results_).loc[:, ['param_C', 'mean_test_score']]
print(results.sort_values(by='mean_test_score', ascending=False))



print()
print('Polynomial kernel:')

clf = GridSearchCV(SVC(kernel='poly'), {'degree': [2, 3, 4], 'C': [1, 10, 100, 500], 'gamma': [0.01, 0.03, 0.05]})
clf.fit(x, y)
results = pd.DataFrame(clf.cv_results_).loc[:, ['param_degree', 'param_C', 'param_gamma', 'mean_test_score']]
print(results.sort_values(by='mean_test_score', ascending=False))














print()
print()
print('Last Part: Finding best models accuracy')


clf = SVC(C=100.0, kernel='poly')
clf.fit(trainX, trainY)
testPredictions = clf.predict(testX)

print('Accuracy: %f' % accuracy_score(y_pred=testPredictions, y_true=testY))
plot_confusion_matrix(clf, testX, testY)
plt.show()





clf = SVC(C=500.0, gamma=0.9, kernel='rbf')
clf.fit(trainX, trainY)

testPredictions = clf.predict(testX)

print('Accuracy: %f' % accuracy_score(y_pred=testPredictions, y_true=testY))
plot_confusion_matrix(clf, testX, testY)
plt.show()
