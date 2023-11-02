import matplotlib.pyplot as plt
import autograd
import autograd.numpy as np
from tqdm import tqdm



# hyperparameters
m1, m2, l1, l2, g = 0.9, 0.95, 0.9, 1.0, 9.81

deltaT = 0.001
totalTime = 10
stepCount = int(totalTime / deltaT)

r = 6

Q = 0
R = 0.1
std = np.repeat(np.sqrt(R), r)

groundTruthX = [np.mean(np.array([10.5, 11, 10.3, 10.8, 11.2, 10.2])[0:r]), np.mean(np.array([5.0, 5.2, 5.3, 4.8, 5.5, 5.1])[0:r])]
xHat_1 = [9.5, 4]
P_1 = [5, 5]

n = len(xHat_1)
















def stateTransition(x, deltaT):
    states = []

    states.append(np.cos(x[0]))
    states.append(-np.sin(x[1]))

    return deltaT * np.array(states)




def measurement(x):
    measurements = []

    measurements.append(x[0] ** 2 + x[1] ** 2)
    measurements.append(np.arctan2(x[1], x[0]))
    measurements.append(x[0] * np.sin(x[1]))
    measurements.append(x[1] * np.cos(x[0]))
    measurements.append(x[0] ** 3 + x[1] ** 3)
    measurements.append(x[0] * x[1] ** 2)

    return np.array(measurements[0:r])




calcJacobianF = autograd.jacobian(stateTransition)
calcJacobianH = autograd.jacobian(measurement)

groundTruthX = np.array(groundTruthX)
xHat_1 = np.array(xHat_1)
P_1 = np.diag(P_1)
groundTruthXAll = []
xHatAll = []







for i in tqdm(range(stepCount)):
    xHat = xHat_1 + stateTransition(xHat_1, deltaT)
    A = calcJacobianF(xHat_1, deltaT)
    P = A @ P_1 @ A.T + Q

    H = calcJacobianH(xHat)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + np.diag(np.random.multivariate_normal(np.zeros(r), np.diag(std))))
    xHat = xHat + K @ (measurement(groundTruthX) - measurement(xHat))
    P = (np.eye(n) - K @ H) @ P

    xHat_1 = xHat
    P_1 = P

    groundTruthX = groundTruthX + stateTransition(groundTruthX, deltaT)

    groundTruthXAll.append(groundTruthX)
    xHatAll.append(xHat)

groundTruthXAll = np.array(groundTruthXAll)
xHatAll = np.array(xHatAll)









steps = np.arange(0, totalTime, deltaT)

plt.figure()
plt.title('Ground Truth vs Predicted x')
plt.plot(steps, groundTruthXAll[:, 0], '-', color='g', label='Ground Truth x')
plt.plot(steps, xHatAll[:, 0], '--', color='r', label='Predicted x')
plt.xlabel('Time Step')
plt.ylabel('x')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.title('Ground Truth vs Predicted y')
plt.plot(steps, groundTruthXAll[:, 1], '-', color='g', label='Ground Truth y')
plt.plot(steps, xHatAll[:, 1], '--', color='r', label='Predicted y')
plt.xlabel('Time Step')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()






plt.figure()
plt.title('Ground Truth vs Predicted x Error')
plt.plot(steps, groundTruthXAll[:, 0] - xHatAll[:, 0], '-', color='g', label='x Error')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.title('Ground Truth vs Predicted y Error')
plt.plot(steps, groundTruthXAll[:, 1] - xHatAll[:, 1], '-', color='g', label='y Error')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.grid()
plt.legend()
plt.show()






print()
print('x MAE: ' + str(np.mean(np.abs(groundTruthXAll[:, 0] - xHatAll[:, 0]))))
print('y MAE: ' + str(np.mean(np.abs(groundTruthXAll[:, 1] - xHatAll[:, 1]))))
