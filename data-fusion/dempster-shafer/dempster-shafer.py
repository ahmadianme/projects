import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyds import MassFunction, powerset
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")





drawPlots = True





def fit(xtrain, ytrain):
    trainAll = xtrain.copy()
    trainAll['class'] = ytrain

    classRange = {}
    for c in trainAll['class'].unique():
        fieldRange = {}
        fields = xtrain.columns

        for f in fields:
            fieldRange[f] =(trainAll[trainAll['class'] == c][f].min(), trainAll[trainAll['class'] == c][f].max())
        classRange[c] = fieldRange

    return classRange






df = pd.read_csv(r'iris.csv',)



xtrain, xtest, ytrain, ytest = train_test_split(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], df[['class']], test_size=0.3)

trainAll = xtrain.copy()
trainAll['class'] = ytrain

testAll = xtest.copy()
testAll['class'] = ytest

print('xtrain length: ' + str(len(xtrain)))
print('xtest length:  ' + str(len(xtest)))
print('ytrain length: ' + str(len(ytrain)))
print('ytest length:  ' + str(len(ytest)))
print()








# part 2 #######################################################################################
versicolor = trainAll.loc[trainAll['class'] == 'Versicolor']
virginica = trainAll.loc[trainAll['class'] == 'Virginica']
setosa = trainAll.loc[trainAll['class'] == 'Setosa']

if drawPlots:
    sns.distplot(versicolor[['sepal_length']], hist=False, label='Versicolor')
    sns.distplot(virginica[['sepal_length']], hist=False, label='Virginica')
    sns.distplot(setosa[['sepal_length']], hist=False, label='Setosa')
    plt.title('Versicolor Class "sepal_length" Density')
    plt.legend()
    plt.show()




versicolor = trainAll.loc[trainAll['class'] == 'Versicolor']
virginica = trainAll.loc[trainAll['class'] == 'Virginica']
setosa = trainAll.loc[trainAll['class'] == 'Setosa']

if drawPlots:
    sns.distplot(versicolor[['sepal_width']], hist=False, label='Versicolor')
    sns.distplot(virginica[['sepal_width']], hist=False, label='Virginica')
    sns.distplot(setosa[['sepal_width']], hist=False, label='Setosa')
    plt.title('Versicolor Class "sepal_width" Density')
    plt.legend()
    plt.show()




versicolor = trainAll.loc[trainAll['class'] == 'Versicolor']
virginica = trainAll.loc[trainAll['class'] == 'Virginica']
setosa = trainAll.loc[trainAll['class'] == 'Setosa']

if drawPlots:
    sns.distplot(versicolor[['petal_length']], hist=False, label='Versicolor')
    sns.distplot(virginica[['petal_length']], hist=False, label='Virginica')
    sns.distplot(setosa[['petal_length']], hist=False, label='Setosa')
    plt.title('Versicolor Class "petal_length" Density')
    plt.legend()
    plt.show()




versicolor = trainAll.loc[trainAll['class'] == 'Versicolor']
virginica = trainAll.loc[trainAll['class'] == 'Virginica']
setosa = trainAll.loc[trainAll['class'] == 'Setosa']

if drawPlots:
    sns.distplot(versicolor[['petal_width']], hist=False, label='Versicolor')
    sns.distplot(virginica[['petal_width']], hist=False, label='Virginica')
    sns.distplot(setosa[['petal_width']], hist=False, label='Setosa')
    plt.title('Versicolor Class "petal_width" Density')
    plt.legend()
    plt.show()



















# part 3 ######################################################################################
print("\n\n\nPart 3 ------------------------------------------------------------------------------------------------------------\n")

omega = trainAll['class'].unique()
powerSetMap = powerset(omega)
powerSet = dict.fromkeys(powerSetMap, 0)
fod = MassFunction(powerSet)

print('Omega:')
print(omega)
print()
print('Power Set:')
pprint(powerSet)
print()
print('FoD:')
pprint(fod)
print()














# part 4 #####################################################################################
print("\n\n\nPart 4 ------------------------------------------------------------------------------------------------------------\n")

classRange = fit(xtrain, ytrain)
print('Class Range (Z):')
pprint(classRange)
print()












# part 5 #####################################################################################
print("\n\n\nPart 5 ------------------------------------------------------------------------------------------------------------\n")

def assign(z, sample):
    features = {'petal_length', 'petal_width', 'sepal_length', 'sepal_width'}
    allClasses = set(trainAll['class'].unique())

    mass = {}

    for feature in features:
        mass[feature] = dict.fromkeys(powerset(allClasses), 0)



    for feature in features:
        classes = set()

        for c in z.keys():
            if classRange[c][feature][0] <= sample[feature] and sample[feature] <= classRange[c][feature][1]:
                classes.add(c)


        if len(classes) == 1:
            members = []

            for member in mass[feature]:
                if len(member) != 0:
                    member = set(member)

                    if list(classes)[0] in member:
                        if member != classes:
                            members.append(member)

            mass[feature][frozenset(classes)] = 1.1 - (0.1 * (len(members) + 1))

            for member in members:
                mass[feature][frozenset(member)] = 0.1



        elif len(classes) == 2:
            mass[feature][frozenset(classes)] = 0.7

            mass[feature][frozenset(allClasses)] = 0.1

            for member in classes:
                mass[feature][frozenset({member})] = 0.1



        else:
            for member in mass[feature]:
                if len(member) == 1:
                    mass[feature][member] = 0.1

            mass[feature][frozenset(allClasses)] = 1 - (0.1 * len(allClasses))

    return mass






def calcK(m1, m2):
    k = 0

    for member1 in m1.keys():
        for member2 in m2.keys():
            if len(member1) == 0 or len(member2) == 0:
                continue

            if len(frozenset.intersection(member1, member2)) == 0:
                k += m1[member1] * m2[member2]


    return k






def calcJointMassNumerator(m1, m2, k):
    jointMassNumerator = {}

    for target in m1.keys():
        sum = 0

        if len(target) == 0:
            jointMassNumerator[target] = sum
            continue

        calculatedTuples = []


        for member1 in m1.keys():
            if len(member1) == 0:
                continue

            for member2 in m2.keys():
                if len(member2) == 0:
                    continue

                if member1 == member2 and ((member1, member2) in calculatedTuples or (member2, member1) in calculatedTuples):
                    continue

                if frozenset.intersection(member1, member2) == target:
                    calculatedTuples.append((member1, member2))
                    sum += m1[member1] * m2[member2]


        jointMassNumerator[target] = sum



    for key in jointMassNumerator.keys():
        jointMassNumerator[key] *= (1 / (1 - k))


    return jointMassNumerator





def fuse(mass):
    features = {'petal_length', 'petal_width', 'sepal_length', 'sepal_width'}
    allClasses = set(trainAll['class'].unique())

    k = calcK(mass['petal_length'], mass['petal_width'])
    jointMass = calcJointMassNumerator(mass['petal_length'], mass['petal_width'], k)
    print(k)

    k = calcK(jointMass, mass['sepal_length'])
    jointMass = calcJointMassNumerator(jointMass, mass['sepal_length'], k)
    print(k)

    k = calcK(jointMass, mass['sepal_width'])
    jointMass = calcJointMassNumerator(jointMass, mass['sepal_width'], k)
    print(k)
    print()

    return jointMass





def predict(jointMass):
    return set(max(jointMass, key=jointMass.get))




def calcAccuracy(classes, y, predictions):
    accuracy = 0
    shallowAccuracy = 0

    classesTotal = dict.fromkeys(classes, 0)
    classesCurrect = dict.fromkeys(classes, 0)

    for i in range(len(predictions)):
        classesTotal[list(y[i])[0]] += 1

        if predictions[i] == y[i]:
            accuracy += 1
            classesCurrect[list(y[i])[0]] += 1

        if len(set.intersection(predictions[i], y[i])) > 0:
            shallowAccuracy += 1

    classesAccuracy = np.array(list(classesCurrect.values())) / np.array(list(classesTotal.values()))
    classesAccuracy = dict(zip(classes, classesAccuracy.tolist()))

    return accuracy / len(predictions), classesAccuracy, shallowAccuracy / len(predictions)






predictions = []

for index, sample in xtest.iterrows():
    mass = assign(classRange, sample)

    jointMass = fuse(mass)

    prediction = predict(jointMass)
    predictions.append(prediction)


classes = df['class'].unique()

y = [{x} for x in ytest['class'].values.tolist()]
accuracy, classesAccuracy, shallowAccuracy = calcAccuracy(classes, y, predictions)
print('Accuracy:         ', accuracy)
print('Shallow Accuracy: ', shallowAccuracy)
print('Classes Accuracy: ')
pprint(classesAccuracy)



testAll['prediction'] = predictions

testAll.to_csv('predictions.csv')
