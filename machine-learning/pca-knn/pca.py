import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def read_images(dirName):
    data = []
    label = []
    for root, dirs, files in os.walk(dirName):
        for file in files:
            face = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
            face = face.reshape(48 * 48, ).tolist()
            data.append(face)

            label.append(root.split('/')[-1])
    return np.asarray(data) , np.asarray(label)





train_data , train_label = read_images('FER2013/train')
test_data , test_label = read_images('FER2013/test')

print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)

print(set(train_label))





fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

idxs = np.random.randint(0, train_data.shape[0], size=4)
for i, idx in enumerate(idxs):
    axs[i].set_title(train_label[idx])
    axs[i].imshow(train_data[idx].reshape(48,48), cmap='gray')
    axs[i].axis('off')

plt.show()





pca = PCA(random_state=123).fit(train_data)

plt.figure(figsize=(20, 6))

plt.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, label='Eigenvalues per component')
plt.xticks(range(1, pca.components_.shape[0]+1, 100))
plt.legend()
plt.show()


cdf = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(20, 6))
plt.plot(range(len(cdf)), cdf, label='Eigenvalues CDF')
plt.xticks(range(1, pca.components_.shape[0]+1, 100))

for limit in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
    idx = np.argmax(cdf>=limit)
    plt.plot(idx, cdf[idx], 'o', label='# of components: %d, Evgeivalues CDF: %0.2f' % (idx, limit))

plt.legend()
plt.show()


fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

for i in range(4):
    axs[i].set_title('Variance: %0.3f' % pca.explained_variance_ratio_[i])
    axs[i].imshow(pca.components_[i].reshape(48, 48), cmap='gray')
    axs[i].axis('off')


fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

for i in range(4):
    axs[i].set_title('Variance: %0.3f' % pca.explained_variance_ratio_[-i-1])
    axs[i].imshow(pca.components_[-i-1].reshape(48, 48), cmap='gray')
    axs[i].axis('off')



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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





nn = KNeighborsClassifier(n_neighbors=1)

nn.fit(train_data, train_label)
predict = nn.predict(test_data)





labels = list(set(test_label))
cm = confusion_matrix(test_label, predict, labels=labels)

plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, labels)





ccr = [cm[i, i] / cm[i, :].sum() for i in range(len(set(test_label)))]
labels = list(set(test_label))

for i, l in enumerate(labels):
    print('CCR-%s: %0.2f' % (l, ccr[i]))




nn = KNeighborsClassifier(n_neighbors=2)

nn.fit(train_data, train_label)
predict = nn.predict(test_data)






labels = list(set(test_label))
cm = confusion_matrix(test_label, predict, labels=labels)

plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, labels)





ccr = [cm[i, i] / cm[i, :].sum() for i in range(len(set(test_label)))]
labels = list(set(test_label))
for i, l in enumerate(labels):
    print('CCR-%s: %0.2f' % (l, ccr[i]))





pca = PCA(random_state=123).fit(train_data)

train_data_trans = pca.transform(train_data)
test_data_trans = pca.transform(test_data)



nn = KNeighborsClassifier(n_neighbors=1)

nn.fit(train_data_trans, train_label)
predict = nn.predict(test_data_trans)






labels = list(set(test_label))
cm = confusion_matrix(test_label, predict, labels=labels)

plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, labels)







ccr = [cm[i, i] / cm[i, :].sum() for i in range(len(set(test_label)))]
labels = list(set(test_label))
for i, l in enumerate(labels):
    print('CCR-%s: %0.2f' % (l, ccr[i]))







pca = PCA(random_state=123).fit(train_data)

train_data_trans = pca.transform(train_data)
test_data_trans = pca.transform(test_data)



nn = KNeighborsClassifier(n_neighbors=2)

nn.fit(train_data_trans, train_label)
predict = nn.predict(test_data_trans)





labels = list(set(test_label))
cm = confusion_matrix(test_label, predict, labels=labels)

plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, labels)






ccr = [cm[i, i] / cm[i, :].sum() for i in range(len(set(test_label)))]
labels = list(set(test_label))
for i, l in enumerate(labels):
    print('CCR-%s: %0.2f' % (l, ccr[i]))






labels = list(set(test_label))

ccrs = {l:[] for l in labels}

components = range(1, 200, 10)
for n in components:
    pca = PCA(n_components=n, random_state=123).fit(train_data)

    train_data_trans = pca.transform(train_data)
    test_data_trans = pca.transform(test_data)

    nn = KNeighborsClassifier(n_neighbors=1)
    nn.fit(train_data_trans, train_label)
    predict = nn.predict(test_data_trans)

    cm = confusion_matrix(test_label, predict, labels=labels)
    ccr = [cm[i, i] / cm[i, :].sum() for i in range(len(set(test_label)))]
    for i, l in enumerate(labels):
        ccrs[l].append(ccr[i])




plt.figure(figsize=(20, 10))

for l in labels:
    plt.plot(components, ccrs[l], label=l)

plt.xticks(components)
plt.legend()
plt.show()






labels = list(set(test_label))

ccrs = {l:[] for l in labels}

components = range(1, 200, 10)
for n in components:
    pca = PCA(n_components=n, random_state=123).fit(train_data)

    train_data_trans = pca.transform(train_data)
    test_data_trans = pca.transform(test_data)

    nn = KNeighborsClassifier(n_neighbors=2)
    nn.fit(train_data_trans, train_label)
    predict = nn.predict(test_data_trans)

    cm = confusion_matrix(test_label, predict, labels=labels)
    ccr = [cm[i, i] / cm[i, :].sum() for i in range(len(set(test_label)))]
    for i, l in enumerate(labels):
        ccrs[l].append(ccr[i])




plt.figure(figsize=(20, 10))

for l in labels:
    plt.plot(components, ccrs[l], label=l)

plt.xticks(components)
plt.legend()
plt.show()

