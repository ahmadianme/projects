import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise



# hyperparameters
m1, m2, l1, l2, g = 0.9, 0.95, 0.9, 1.0, 9.81

deltaT = 0.001
totalTime = 10
stepCount = int(totalTime / deltaT)

Q = np.diag([0.1, 0.12, 0.05, 0.03])
R = np.diag([0.2, 0.5])

groundTruthX = [0.5 * np.pi, 0.5 * np.pi, 0.0, 0.0]
xHat_1 = [0.4 * np.pi, 0.4 * np.pi, 0.1 * np.pi, 0.1 * np.pi]
P_1 = [1, 1, 1, 1]

n = len(xHat_1)
r = 2















def stateTransition(x, deltaT):
    states = []

    states.append(x[2])
    states.append(x[3])
    states.append((-g * (2 * m1 + m2) * np.sin(x[0]) - m2 * g * np.sin(x[0] - 2 * x[1]) - 2 * np.sin(x[0] - x[1]) * m2 * (x[3] ** 2 * l2 + x[2] ** 2 * l1 * np.cos(x[0] - x[1]))) / (l1 * (2 * m1 + m2 - m2 * np.cos(2 * x[0] - 2 * x[1]))))
    states.append((2 * np.sin(x[0] - x[1]) * (x[2] ** 2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(x[0]) + x[3] ** 2 * l2 * m2 * np.cos(x[0] - x[1]))) / (l2 * (2 * m1 + m2 - m2 * np.cos(2 * x[0] - 2 * x[1]))))

    return deltaT * np.array(states)




def measurement(x):
    return x[0:r]





points = MerweScaledSigmaPoints(n=n, alpha=0.1, beta=2.0, kappa=3-n)
ukf = UnscentedKalmanFilter(dim_x=n, dim_z=r, dt=deltaT, fx=stateTransition, hx=measurement, points=points)

ukf.Q = Q
ukf.R = R

groundTruthX = np.array(groundTruthX)
ukf.x = np.array(xHat_1)
ukf.P = np.diag(P_1)
groundTruthXAll = []
xHatAll = []






for step in tqdm(range(stepCount)):
    ukf.predict()

    z = measurement(groundTruthX)
    ukf.update(z)

    xHat_1 = ukf.x.copy() + groundTruthX

    groundTruthX = groundTruthX + stateTransition(groundTruthX, deltaT)

    groundTruthXAll.append(groundTruthX)
    xHatAll.append(xHat_1)

groundTruthXAll = np.array(groundTruthXAll)
xHatAll = np.array(xHatAll)









steps = np.arange(0, totalTime, deltaT)

plt.figure()
plt.title('Ground Truth vs Predicted Theta1')
plt.plot(steps, groundTruthXAll[:, 0], '-', color='g', label='Ground Truth Theta1')
plt.plot(steps, xHatAll[:, 0], '--', color='r', label='Predicted Theta1')
plt.xlabel('Time Step')
plt.ylabel('Theta1')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.title('Ground Truth vs Predicted Theta2')
plt.plot(steps, groundTruthXAll[:, 1], '-', color='g', label='Ground Truth Theta2')
plt.plot(steps, xHatAll[:, 1], '--', color='r', label='Predicted Theta2')
plt.xlabel('Time Step')
plt.ylabel('Theta2')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.title('Ground Truth vs Predicted Omega1')
plt.plot(steps, groundTruthXAll[:, 2], '-', color='g', label='Ground Truth Omega1')
plt.plot(steps, xHatAll[:, 2], '--', color='r', label='Predicted Omega1')
plt.xlabel('Time Step')
plt.ylabel('Omega1')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.title('Ground Truth vs Predicted Omega2')
plt.plot(steps, groundTruthXAll[:, 3], '-', color='g', label='Ground Truth Omega2')
plt.plot(steps, xHatAll[:, 3], '--', color='r', label='Predicted Omega2')
plt.xlabel('Time Step')
plt.ylabel('Omega2')
plt.grid()
plt.legend()
plt.show()







plt.figure()
plt.title('Ground Truth vs Predicted Theta1 Error')
plt.plot(steps, groundTruthXAll[:, 0] - xHatAll[:, 0], '-', color='g', label='Theta1 Error')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.title('Ground Truth vs Predicted Theta2 Error')
plt.plot(steps, groundTruthXAll[:, 1] - xHatAll[:, 1], '-', color='g', label='Theta2 Error')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.title('Ground Truth vs Predicted Omega1 Error')
plt.plot(steps, groundTruthXAll[:, 2] - xHatAll[:, 2], '-', color='g', label='Omega1 Error')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.title('Ground Truth vs Predicted Omega2 Error')
plt.plot(steps, groundTruthXAll[:, 3] - xHatAll[:, 3], '-', color='g', label='Omega2 Error')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.grid()
plt.legend()
plt.show()







print()
print('Theta1 MAE: ' + str(np.mean(np.abs(groundTruthXAll[:, 0] - xHatAll[:, 0]))))
print('Theta2 MAE: ' + str(np.mean(np.abs(groundTruthXAll[:, 1] - xHatAll[:, 1]))))
print('Omega1 MAE: ' + str(np.mean(np.abs(groundTruthXAll[:, 2] - xHatAll[:, 2]))))
print('Omega2 MAE: ' + str(np.mean(np.abs(groundTruthXAll[:, 3] - xHatAll[:, 3]))))
