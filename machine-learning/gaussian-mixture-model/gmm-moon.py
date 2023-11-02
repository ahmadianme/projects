import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets




moonsDataset = datasets.make_moons(n_samples=500, noise=0.11)
X = moonsDataset[0]
y = moonsDataset[1]




plt.figure(figsize=(10, 8))
plt.scatter(X[y==0, 0], X[y==0, 1], color='purple', label='Top Moon')
plt.scatter(X[y==1, 0], X[y==1, 1], color='green', label='Bottom Moon')

plt.legend()

plt.tight_layout()




X0 = X[y==0]
mu0 = np.mean(X[y==0], axis=0)
sig0 = np.cov(X[y==0].T)

X1 = X[y==1]
mu1 = np.mean(X[y==1], axis=0)
sig1 = np.cov(X[y==1].T)

print(mu0)
print(sig0)

print(mu1)
print(sig1)




def multivariate_gaussian(mu, sigma, X):
    diff = X - mu[:, np.newaxis]
    det_sigma = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    g = (np.exp(-(diff.T @ inv_sigma @ diff) / 2)) / (2 * np.pi * np.sqrt(det_sigma))
    return np.diag(g)



X_mesh, Y_mesh = np.meshgrid(np.linspace(-1.5, 2.5), np.linspace(-1, 1.5))
XY_mesh = np.array([X_mesh.ravel(), Y_mesh.ravel()])

plt.figure(figsize=(10, 8))
plt.scatter(X0[:, 0], X0[:, 1], color='purple', label='Top Moon', alpha=0.25)
plt.scatter(X1[:, 0], X1[:, 1], color='green', label='Bottom Moon', alpha=0.25)

Z0 = multivariate_gaussian(mu0, sig0, XY_mesh).reshape(X_mesh.shape)
Z1 = multivariate_gaussian(mu1, sig1, XY_mesh).reshape(X_mesh.shape)

plt.contour(X_mesh, Y_mesh, Z0, levels=5, colors='purple')
plt.contour(X_mesh, Y_mesh, Z1, levels=5, colors='green')

plt.legend()

plt.tight_layout()



def multivariate_gaussian(mu, sigma, X):
    diff = X - mu[:, np.newaxis]
    det_sigma = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    g = (np.exp(-(diff.T @ inv_sigma @ diff) / 2)) / (2 * np.pi * np.sqrt(det_sigma))
    return np.diag(g)



class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=25, reg_covar=1e-6):
        np.random.seed(0)
        self.n_components = n_components
        self.max_iter = int(max_iter)
        self.reg_covar = reg_covar

    def initVars(self, X):
        self.shape = X.shape
        self.n, self.n_features = self.shape
        self.alpha = np.full(shape=self.n_components, fill_value=(1/self.n_components))
        random_row = np.random.randint(low=0, high=self.n, size=self.n_components)
        self.mu = [X[row_index,:] for row_index in random_row]
        self.sigma = [np.cov(X.T) for _ in range(self.n_components)]

    def do_e_step(self, X):
        self.weights = self.predict_probabilities(X)
        self.alpha = self.weights.mean(axis=0)

    def do_m_step(self, X):
        for j in range(self.n_components):
            weight = self.weights[:, j].reshape((self.n, 1))
            total_weight = weight.sum()
            self.mu[j] = (X * weight).sum(axis=0) / total_weight
            self.sigma[j] = np.cov(X.T, aweights=(weight/total_weight).flatten(), bias=True)
            self.sigma[j].flat[::(self.n_features+1)] += self.reg_covar

    def fit(self, X):
        self.initVars(X)

        for iteration in range(self.max_iter):
            self.do_e_step(X)
            self.do_m_step(X)

    def predict_probabilities(self, X):
        likelihood = np.zeros((self.n, self.n_components))
        for j in range(self.n_components):
            likelihood[:, j] = multivariate_gaussian(self.mu[j], self.sigma[j], X.T)

        return (self.alpha * likelihood) / (self.alpha * likelihood).sum(axis=1)[:, np.newaxis]

    def score_samples(self, X):
        scores = np.zeros(X.shape[0])
        for i in range(self.n_components):
            scores += self.alpha[i] * multivariate_gaussian(self.mu[i], self.sigma[i], X.T)
        return scores

    def _n_parameters(self):
        cov_params = self.n_components * self.n_features * (self.n_features + 1) / 2.
        mean_params = self.n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def calculateAIC(self, X):
        return -2 * self.score_samples(X).mean() * X.shape[0] + 2 * self._n_parameters()

    def calculateBIC(self, X):
        return (-2 * self.score_samples(X).mean() * X.shape[0] + self._n_parameters() * np.log(X.shape[0]))

    def print_report(self):
        for idx in range(self.n_components):
            print('\u03B1_%d:'%idx)
            print(self.alpha[idx])
            print('\u03BC_%d:'%idx)
            print(self.mu[idx])
            print('\u03A3_%d:'%idx)
            print(self.sigma[idx])
            print()




fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 24), sharex=True)

X_mesh, Y_mesh = np.meshgrid(np.linspace(-1.5, 2.5), np.linspace(-1, 1.5))
XY_mesh = np.array([X_mesh.ravel(), Y_mesh.ravel()]).T

aics0 = []
bics0 = []

aics1 = []
bics1 = []

idx = 0
for nc in range(1, 17):
    gmm0 = GaussianMixtureModel(nc)
    gmm0.fit(X0)
    aics0.append(gmm0.calculateAIC(X0))
    bics0.append(gmm0.calculateBIC(X0))

    gmm1 = GMM(nc)
    gmm1.fit(X1)
    aics1.append(gmm1.aic(calculateAIC))
    bics1.append(gmm1.calculateBIC(X1))

    if nc in (3, 8, 16):
        axs[idx].set_title('%d components for each class' % nc)

        axs[idx].scatter(X0[:, 0], X0[:, 1], color='purple', label='Top Moon', alpha=0.25)
        axs[idx].scatter(X1[:, 0], X1[:, 1], color='green', label='Bottom Moon', alpha=0.25)

        Z0 = gmm0.score_samples(XY_mesh).reshape(X_mesh.shape)
        axs[idx].contour(X_mesh, Y_mesh, Z0, colors='purple')

        Z1 = gmm1.score_samples(XY_mesh).reshape(X_mesh.shape)
        axs[idx].contour(X_mesh, Y_mesh, Z1, colors='green')

        axs[idx].legend()

        idx += 1

plt.tight_layout()


# In[17]:


fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))
ncs = range(1, 17)

axs[0].plot(ncs, aics0, 'bo-', label='Too Moon AIC')
axs[0].plot(ncs, aics1, 'go-', label='Bottom Moon AIC')

axs[0].set_xticks(ncs)
axs[0].legend()

axs[1].plot(ncs, bics0, 'bo-', label='Top Moon BIC')
axs[1].plot(ncs, bics1, 'go-', label='Bottom Moon BIC')

axs[1].set_xticks(ncs)
axs[1].legend()

plt.tight_layout()


# ## Best Model
