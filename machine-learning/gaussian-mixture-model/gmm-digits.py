import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture







digits = load_digits()
print(digits.data.shape)

X = digits.data
y = digits.target






aics = {c: [] for c in range(10)}
for nc in range(1, 30):
    for c in range(10):
        gm = GaussianMixture(n_components=nc, random_state=123).fit(X[y==c])
        aics[c].append(gm.aic(X[y==c]))




plt.figure(figsize=(20, 10))

for c in range(10):
    plt.plot(range(1, 30), aics[c], label='Class: ' + str(c))
    idx = np.argmin(aics[c])
    plt.plot(idx+1, aics[c][idx], 'bx')
plt.xticks(range(1, 30))

plt.legend()
plt.show()




gms = {c: None for c in range(10)}
for c in range(10):
    gms[c] = GaussianMixture(n_components=11, random_state=123).fit(X[y==c])



fig, axs = plt.subplots(10, 10, figsize=(10,10))
for c in range(10):
    samples = gms[c].sample(10)[0]
    for i in range(10):
        axs[c][i].imshow(samples[i].reshape(8, 8))
        axs[c][i].set_xticks([])
        axs[c][i].set_yticks([])






pca = PCA(random_state=123).fit(X)

plt.figure(figsize=(16, 8))
cdf = np.cumsum(pca.explained_variance_ratio_)

plt.plot(range(len(cdf)), cdf, label='Eivenvalues CDF')
plt.xticks(range(1, pca.components_.shape[0]+1, 9))

for limit in [0.5, 0.75, 0.85, 0.9, 0.95, 0.99]:
    idx = np.argmax(cdf>=limit)
    plt.plot(idx, cdf[idx], 'x', label='# of components: %d, Eivenvalues: %0.2f' % (idx, limit))

plt.legend()
plt.show()




pca = PCA(n_components=40, random_state=123).fit(X)
X_trans = pca.transform(X)






aics = {c: [] for c in range(10)}
for nc in range(1, 30):
    for c in range(10):
        gm = GaussianMixture(n_components=nc, random_state=123).fit(X_trans[y==c])
        aics[c].append(gm.aic(X_trans[y==c]))




plt.figure(figsize=(16, 8))

for c in range(10):
    plt.plot(range(1, 30), aics[c], label='Class: ' + str(c))
    idx = np.argmin(aics[c])
    plt.plot(idx+1, aics[c][idx], 'bx')
plt.xticks(range(1, 30))

plt.legend()
plt.show()




gms = {c: None for c in range(10)}
for c in range(10):
    gms[c] = GaussianMixture(n_components=18, random_state=123).fit(X_trans[y==c])




fig, axs = plt.subplots(10, 10, figsize=(10,10))
for c in range(10):
    samples = gms[c].sample(10)[0]
    samples = pca.inverse_transform(samples)
    for i in range(10):
        axs[c][i].imshow(samples[i].reshape(8, 8))
        axs[c][i].set_xticks([])
        axs[c][i].set_yticks([])
