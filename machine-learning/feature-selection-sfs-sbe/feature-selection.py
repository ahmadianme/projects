import itertools
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB





with open('madelon/madelon_train.data') as f:
    lines = f.readlines()
    X_train = np.array([line.strip().split(' ') for line in lines], dtype=np.int16)
y_train = np.loadtxt('madelon/madelon_train.labels', dtype=np.int8, delimiter=' ')


with open('madelon/madelon_valid.data') as f:
    lines = f.readlines()
    X_val = np.array([line.strip().split(' ') for line in lines], dtype=np.int16)
y_val = np.loadtxt('madelon/madelon_valid.labels', dtype=np.int8, delimiter=' ')

print(X_train.shape)
print(X_val.shape)



def draw_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')






def objective_function(feature_subset):
    clf = GaussianNB(var_smoothing=1e-5)
    cv = cross_val_score(clf, X_train[:, feature_subset], y_train, cv=3, scoring='neg_mean_squared_error')
    return cv.mean()

# forward selection
def forward_selection():
    feature_set = list(range(X_train.shape[1]))
    feature_subset = []
    neg_mean_squared_errors = []
    for _ in range(X_train.shape[1]):
        objs = [objective_function(feature_subset+[feature]) for feature in feature_set if feature not in feature_subset]
        idx = np.argmax(objs)
        neg_mean_squared_errors.append(objs[idx])
        feature_subset.append(feature_set[idx])
        feature_set.remove(feature_set[idx])

    return feature_subset, neg_mean_squared_errors


start_time = time.time()

feature_subset, neg_mean_squared_errors = forward_selection()

print('Time: ' + str(time.time() - start_time))

"""#### Optimal Feature Subset"""

plt.figure(figsize=(16, 8))
plt.plot(neg_mean_squared_errors)
plt.show()

m_idx = np.argmax(neg_mean_squared_errors)


ofs = feature_subset[:m_idx]
print(ofs)



clf = GaussianNB(var_smoothing=1e-5)
clf.fit(X_train[:, ofs], y_train)

y_predict = clf.predict(X_val[:, ofs])


cm = confusion_matrix(y_val, y_predict)

plt.figure(figsize=(10, 10))
draw_confusion_matrix(cm, [-1, 1])


print(accuracy_score(y_val, y_predict))






def objective_function(feature_subset):
    clf = GaussianNB(var_smoothing=1e-5)
    cv = cross_val_score(clf, X_train[:, feature_subset], y_train, cv=3, scoring='neg_mean_squared_error')
    return cv.mean()

# backward elimination
def backward_elimination():
    feature_subset = list(range(X_train.shape[1]))
    neg_mean_squared_errors = []
    for _ in range(X_train.shape[1]):
        objs = []
        for feature in feature_subset:
            feature_subset_copy = feature_subset.copy()
            feature_subset_copy.remove(feature)
            obj = objective_function(feature_subset_copy)
            objs.append(obj)

        idx = np.argmax(objs)
        neg_mean_squared_errors.append(objs[idx])
        feature_subset.remove(feature_subset[idx])

    return feature_subset, neg_mean_squared_errors

feature_subset, neg_mean_squared_errors = backward_elimination()

print('Time: ' + str(time.time() - start_time))


plt.figure(figsize=(20, 9))
plt.plot(neg_mean_squared_errors)
plt.show()

m_idx = np.argmax(neg_mean_squared_errors)

ofs = feature_subset[:m_idx]
print(ofs)



clf = GaussianNB(var_smoothing=1e-5)
clf.fit(X_train[:, ofs], y_train)

y_predict = clf.predict(X_val[:, ofs])


cm = confusion_matrix(y_val, y_predict)

plt.figure(figsize=(10, 10))
draw_confusion_matrix(cm, [-1, 1])


print(accuracy_score(y_val, y_predict))
