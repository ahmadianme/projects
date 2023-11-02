import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# warnings.simplefilter("ignore")





df = pd.read_excel(r'Data.xlsx', header=None)
# print(df[0])
# print(df[1])
# print(df[2])
# print(len(df.iloc[:, : 3].values.tolist()))
dataActual = df[3].values.tolist()
dataS1 = df[0].values.tolist()
dataS2 = df[1].values.tolist()
dataS3 = df[2].values.tolist()
dataS = [dataS1, dataS2, dataS3]




def mae(y, yp):
    sum = 0

    for i in range(len(y)):
        sum += abs(y[i] - yp[i])

    return sum / len(y)




def mse(y, yp):
    sum = 0

    for i in range(len(y)):
        sum += np.square(y[i] - yp[i])

    return sum / len(y)





def rmse(y, yp):
    sum = 0

    for i in range(len(y)):
        sum += np.square(y[i] - yp[i])

    return np.sqrt(sum / len(y))





def calcE(y, yp):
    E = []

    for i in range(len(y)):
        E.append(y[i] - yp[i])

    return E



def plotHistogram(E):
    plt.hist(E, bins=50, edgecolor='black')
    plt.title('Histogram Plot')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.show()



def plotKDE(E):
    sns.kdeplot(E, shade=True)
    plt.title('Kernel Density Estimation Plot')
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.show()




def plotPart4(alphas, allOptimisticOrness, allOptimisticDispersion, allMaeOptimistic, allRmseOptimistic):
    plt.plot(alphas, allOptimisticOrness)
    plt.title('Alpha / Orness Plot')
    plt.xlabel('Alpha')
    plt.ylabel('Orness')
    plt.show()

    plt.plot(alphas, allOptimisticDispersion)
    plt.title('Alpha / Dispersion Plot')
    plt.xlabel('Alpha')
    plt.ylabel('Dispersion')
    plt.show()

    plt.plot(alphas, allMaeOptimistic)
    plt.title('Alpha / MAE Plot')
    plt.xlabel('Alpha')
    plt.ylabel('MAE')
    plt.show()

    plt.plot(alphas, allRmseOptimistic)
    plt.title('Alpha / RMSE Plot')
    plt.xlabel('Alpha')
    plt.ylabel('RMSE')
    plt.show()




def calcOrness(w):
    n = len(w)

    sum = 0

    for i in range(1, n + 1):
        sum += (n-i) * w[i-1]

    return 1 / (n-1) * sum



def calcDispersion(w):
    n = len(w)

    sum = 0

    for i in range(1, n + 1):
        if w[i-1] <= 0:
            continue

        sum += w[i-1] * np.log(w[i-1])

    if sum == 0:
        return 0

    return -sum



def calcOptimisticWeights(n, alpha):
    w = []

    for i in range(n - 1):
        w.append(alpha * (np.power(1-alpha, i)))

    w.append(np.power(1-alpha, n - 1))

    return w



def calcPessimisticWeights(n, alpha):
    w = [np.power(alpha, n-1)]

    for i in range(1, n - 1):
        w.append((1-alpha) * np.power(alpha, n-(i+1)))

    w.append(1-alpha)

    return w



def calcObservationalWeights(data, dataActual):
    w = np.repeat(.33, len(data)).tolist()
    stepLambda = np.zeros(len(data)).tolist()

    lr = .35

    for k in range(len(dataActual)):
        dataOrdered = [data[0][k], data[1][k], data[2][k]]
        dataOrdered.sort(reverse=True)

        dHat = np.dot(w, dataOrdered)

        for i in range(len(data)):
            derivative = w[i] * (dataOrdered[i] - dHat) * (dHat - dataActual[k])
            stepLambda[i] = stepLambda[i] - lr * (dataOrdered[i] - dHat) * (dHat - dataActual[k])

        exponentials = np.exp(stepLambda)
        exponentialsSum = np.sum(exponentials)

        for i in range(len(data)):
            w[i] = exponentials[i] / exponentialsSum

    return w



def dependentFusion(data, dataActual):
    allW = []
    allOrness = []
    allDispersion = []
    allFused = []

    for i in range(len(dataActual)):
        dataOrdered = [data[0][i], data[1][i], data[2][i]]
        dataOrdered.sort(reverse=True)

        mu = np.mean(dataOrdered)

        diffSum = np.sum(np.abs(dataOrdered - mu))
        allSim = []

        for j in range(len(dataOrdered)):
            allSim.append(1 - ((np.abs(dataOrdered[j] - mu)) / (diffSum)))

        simSum = np.sum(allSim)

        w = []

        for j in range(len(dataOrdered)):
            w.append(allSim[j] / simSum)

        allW.append(w)

        allOrness.append(calcOrness(w))
        allDispersion.append(calcDispersion(w))

        allFused.append(np.sum(np.dot(allSim, dataOrdered)) / simSum)

    return np.mean(allW, axis=0), np.mean(allOrness), np.mean(allDispersion), allFused





def calcOrLikeSOWAWeights(n, alpha):
    w = [(1/n) * (1-alpha) + alpha]

    for i in range(1, n):
        w.append((1/n) * (1-alpha))

    return w




def windowFusion(data, dataActual, alpha):
    allW = []
    allOrness = []
    allDispersion = []
    allFused = []

    for i in range(len(dataActual)):
        dataOrdered = [data[0][i], data[1][i], data[2][i]]
        dataOrdered.sort(reverse=True)

        dataOrderedPowered = np.sign(dataOrdered) * np.power(np.abs(dataOrdered), alpha)
        dataOrderedPoweredSum = np.sum(dataOrderedPowered)

        dataOrderedPoweredPlus = np.sign(dataOrdered) * np.power(np.abs(dataOrdered), alpha+1)
        dataOrderedPoweredPlusSum = np.sum(dataOrderedPoweredPlus)

        w = []

        for j in range(len(dataOrdered)):
            w.append(dataOrderedPowered[j] / dataOrderedPoweredSum)

        allW.append(w)

        allOrness.append(calcOrness(w))
        allDispersion.append(calcDispersion(w))

        allFused.append(dataOrderedPoweredPlusSum / dataOrderedPoweredSum)

    return np.mean(allW, axis=0), np.mean(allOrness), np.mean(allDispersion), allFused





def fuse(data, w):
    fused = []

    for i in range(len(data[0])):
        dataOrdered = [data[0][i], data[1][i], data[2][i]]
        dataOrdered.sort(reverse=True)

        fused.append(np.dot(w, dataOrdered))

    return fused














# part 1 ################################################################################
mae1 = mae(dataActual, dataS1)
mae2 = mae(dataActual, dataS2)
mae3 = mae(dataActual, dataS3)

mse1 = mse(dataActual, dataS1)
mse2 = mse(dataActual, dataS2)
mse3 = mse(dataActual, dataS3)

rmse1 = rmse(dataActual, dataS1)
rmse2 = rmse(dataActual, dataS2)
rmse3 = rmse(dataActual, dataS3)


print('MAE')
print(mae1)
print(mae2)
print(mae3)
print()
print('MSE')
print(mse1)
print(mse2)
print(mse3)
print()
print('RMSE')
print(rmse1)
print(rmse2)
print(rmse3)
print()










# part 2 ################################################################################
E1 = calcE(dataActual, dataS1)
E2 = calcE(dataActual, dataS2)
E3 = calcE(dataActual, dataS3)

plotHistogram(E1)
plotHistogram(E2)
plotHistogram(E3)

plotKDE(E1)
plotKDE(E2)
plotKDE(E3)









# part 3 ################################################################################
alpha = .6

optimisticWeights = calcOptimisticWeights(3, alpha)
pessimistiWeights = calcPessimisticWeights(3, alpha)

optimisticOrness = calcOrness(optimisticWeights)
optimisticDispersion = calcDispersion(optimisticWeights)
pessimisticOrness = calcOrness(pessimistiWeights)
pessimisticDispersion = calcDispersion(pessimistiWeights)

optimisticFusion = fuse(dataS, optimisticWeights)
pessimisticFusion = fuse(dataS, pessimistiWeights)

# print(optimisticFusion)
# print(len(optimisticFusion))
# print(optimisticOrness)
# print(optimisticDispersion)
# print()

# print(pessimisticFusion)
# print(len(pessimisticFusion))
# print(pessimisticOrness)
# print(pessimisticDispersion)
# print()

maeOptimistic = mae(dataActual, optimisticFusion)
mseOptimistic = mse(dataActual, optimisticFusion)
rmseOptimistic = rmse(dataActual, optimisticFusion)

maePessimistic = mae(dataActual, pessimisticFusion)
msePessimistic = mse(dataActual, pessimisticFusion)
rmsePessimistic = rmse(dataActual, pessimisticFusion)

print('Optimistic Weights: ' + str(optimisticWeights))
print('Pessimistic Weights: ' + str(pessimistiWeights))
print()

print('Optimistic Orness: ', str(optimisticOrness))
print('Optimistic Dispersion: ', str(optimisticDispersion))
print('Pessimistic Orness: ', str(pessimisticOrness))
print('Pessimistic Dispersion: ', str(pessimisticDispersion))
print()

print('MAE: ' + str(maeOptimistic))
print('MSE: ' + str(mseOptimistic))
print('RMSE: ' + str(rmseOptimistic))
print()

print('MAE: ' + str(maePessimistic))
print('MSE: ' + str(msePessimistic))
print('RMSE: ' + str(rmsePessimistic))
print()



EOptimistic = calcE(dataActual, optimisticFusion)
EPessimistic = calcE(dataActual, pessimisticFusion)

plotHistogram(EOptimistic)
plotHistogram(EPessimistic)

plotKDE(EOptimistic)
plotKDE(EPessimistic)














# part 4 ################################################################################
alphas = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

allOptimisticOrness = []
allOptimisticDispersion = []
allMaeOptimistic = []
allRmseOptimistic = []

for alpha in alphas:
    optimisticWeights = calcOptimisticWeights(len(dataS), alpha)

    allOptimisticOrness.append(calcOrness(optimisticWeights))
    allOptimisticDispersion.append(calcDispersion(optimisticWeights))

    optimisticFusion = fuse(dataS, optimisticWeights)

    # print(optimisticFusion)
    # print(len(optimisticFusion))
    # print(optimisticOrness)
    # print(optimisticDispersion)
    # print()

    # print(pessimisticFusion)
    # print(len(pessimisticFusion))
    # print(pessimisticOrness)
    # print(pessimisticDispersion)
    # print()

    allMaeOptimistic.append(mae(dataActual, optimisticFusion))
    allRmseOptimistic.append(rmse(dataActual, optimisticFusion))

plotPart4(alphas, allOptimisticOrness, allOptimisticDispersion, allMaeOptimistic, allRmseOptimistic)

print('Alphas: ' + str(alphas))
print()
print('Orness: ' + str(allOptimisticOrness))
print()
print('Dispersion: ' + str(allOptimisticDispersion))
print()
print('MAE: ' + str(allMaeOptimistic))
print()
print('RMSE: ' + str(allRmseOptimistic))
print()









# # part 5 ################################################################################
observationalWeights = calcObservationalWeights(dataS, dataActual)
observationalOrness = calcOrness(observationalWeights)
observationalDispersion = calcDispersion(observationalWeights)
observationalFusion = fuse(dataS, observationalWeights)

dependentWeights, dependentOrness, dependentDispersion, dependentFusion = dependentFusion(dataS, dataActual)

orLikeSOWAWeights = calcOrLikeSOWAWeights(3, .5)
orLikeSOWAOrness = calcOrness(orLikeSOWAWeights)
orLikeSOWADispersion = calcDispersion(orLikeSOWAWeights)
orLikeSOWAFusion = fuse(dataS, orLikeSOWAWeights)

windowWeights, windowOrness, windowDispersion, windowFusion = windowFusion(dataS, dataActual, .5)

# print(windowFusion)

# EObservational = calcE(dataActual, observationalFusion)
mseObservational = mae(dataActual, observationalFusion)
maeObservational = mse(dataActual, observationalFusion)
rmseObservational = rmse(dataActual, observationalFusion)
print('Observational Weights: ' + str(observationalWeights))
print('Observational Orness: ' + str(observationalOrness))
print('Observational Dispersion: ' + str(observationalDispersion))
print('Observational MAE: ' + str(mseObservational))
print('Observational MSE: ' + str(maeObservational))
print('Observational RMSE: ' + str(rmseObservational))
print()
# plotHistogram(observationalFusion)
# plotKDE(observationalFusion)


# EDependent = calcE(dataActual, dependentFusion)
mseDependent = mae(dataActual, dependentFusion)
maeDependent = mse(dataActual, dependentFusion)
rmseDependent = rmse(dataActual, dependentFusion)
print('Dependent Weights: ' + str(dependentWeights))
print('Dependent Orness: ' + str(dependentOrness))
print('Dependent Dispersion: ' + str(dependentDispersion))
print('Dependent MAE: ' + str(mseDependent))
print('Dependent MSE: ' + str(maeDependent))
print('Dependent RMSE: ' + str(rmseDependent))
print()
# plotHistogram(dependentFusion)
# plotKDE(dependentFusion)

# EOrLikeSOWA = calcE(dataActual, orLikeSOWAFusion)
mseOrLikeSOWA = mae(dataActual, orLikeSOWAFusion)
maeOrLikeSOWA = mse(dataActual, orLikeSOWAFusion)
rmseOrLikeSOWA = rmse(dataActual, orLikeSOWAFusion)
print('OrLikeSOWA Weights: ' + str(orLikeSOWAWeights))
print('OrLikeSOWA Orness: ' + str(orLikeSOWAOrness))
print('OrLikeSOWA Dispersion: ' + str(orLikeSOWADispersion))
print('OrLikeSOWA MAE: ' + str(mseOrLikeSOWA))
print('OrLikeSOWA MSE: ' + str(maeOrLikeSOWA))
print('OrLikeSOWA RMSE: ' + str(rmseOrLikeSOWA))
print()
# plotHistogram(orLikeSOWAFusion)
# plotKDE(orLikeSOWAFusion)

# EWindow = calcE(dataActual, windowFusion)
mseWindow = mae(dataActual, windowFusion)
maeWindow = mse(dataActual, windowFusion)
rmseWindow = rmse(dataActual, windowFusion)

print('Window Weights: ' + str(windowWeights))
print('Window Orness: ' + str(windowOrness))
print('Window Dispersion: ' + str(windowDispersion))
print('Window MAE: ' + str(mseWindow))
print('Window MSE: ' + str(maeWindow))
print('Window RMSE: ' + str(rmseWindow))
print()
# plotHistogram(windowFusion)
# plotKDE(windowFusion)
