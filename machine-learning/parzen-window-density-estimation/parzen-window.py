import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn.neighbors import KernelDensity






data = pd.read_csv('data/ted_main.csv')
samples = data[['duration']]
samples = samples.to_numpy()







def kernel(h, x, xi):
    return (1.0/(np.sqrt(2*np.pi) * h)) * np.exp(-(1.0/2.0)*((x-xi)/h)**2)

def parzen(samples, x, h):
    scores = []

    for sample in samples:
        scores.append(kernel(h=h, x=x, xi=sample))
    return np.mean(scores)




parzen10 = [parzen(samples, x, 10.0) for x in samples.flatten()]
samples10 = np.array([np.random.normal(random.choices(samples.flatten(), parzen10), 10.0) for _ in range(int(1e2))])




plt.figure()
plt.figure(figsize=(8, 5))

sns.distplot(samples)
sns.distplot(samples10, hist=False, label='Window Size: 10')

plt.legend()
plt.show()







parzen20 = [parzen(samples, x, 20.0) for x in samples.flatten()]
samples20 = np.array([np.random.normal(random.choices(samples.flatten(), parzen20), 20.0) for _ in range(int(1e2))])

parzen50 = [parzen(samples, x, 50.0) for x in samples.flatten()]
samples50 = np.array([np.random.normal(random.choices(samples.flatten(), parzen50), 50.0) for _ in range(int(1e2))])

parzen100 = [parzen(samples, x, 100.0) for x in samples.flatten()]
samples100 = np.array([np.random.normal(random.choices(samples.flatten(), parzen100), 100.0) for _ in range(int(1e2))])





plt.figure()
plt.figure(figsize=(8, 5))

sns.distplot(samples)
sns.distplot(samples10, hist=False, label='Window Size: 10')
sns.distplot(samples20, hist=False, label='Window Size: 20')
sns.distplot(samples50, hist=False, label='Window Size: 50')
sns.distplot(samples100, hist=False, label='Window Size: 100')

plt.legend()
plt.show()




np.random.shuffle(samples)

sampleCounts = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
scores = []

sns.distplot(samples)

for sampleCount in sampleCounts:
    density = KernelDensity(kernel='gaussian', bandwidth=10.0).fit(samples[:sampleCount])
    sns.distplot(density.sample(100), hist=False, label="Count: " + str(sampleCount))
    scores.append(density.score(samples[sampleCount:]))

plt.legend()
plt.show()






plt.figure(figsize=(8, 5))
plt.plot(sampleCounts, scores)
plt.xticks(sampleCounts)
plt.xlabel('Sample Count')
plt.ylabel('Total Score')
plt.show()




kde = KernelDensity(kernel='gaussian', bandwidth=10.0).fit(samples)

plt.figure(figsize=(8, 5))
sns.distplot(samples)
sns.distplot(kde.sample(100), hist=False)
plt.show()
