import numpy as np
import random
import math

noiseRate = 0
mistakeRate = 0
testCount = 1000

sp = np.array([
    [1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1],
    [1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1],
])

sn = len(sp)
sm = len(sp[0])

identity = np.identity(sm)

print('Training...')
print()

w = np.zeros((sm, sm))

for p in range(len(sp)):
    w = np.add(w, np.outer(sp[p], np.transpose(sp[p])))


print('Weights Matrix:')
print(w)
print()

w = np.subtract(w, sn * identity)

print('Weights Matrix - PI:')
print(w)
print()


print('Testing ' + str(testCount) + ' cases...')

if (noiseRate > 0):
    print('Adding ' + str(noiseRate*100) + '% noise to input data')
    print()


if (mistakeRate > 0):
    print('Adding ' + str(mistakeRate*100) + '% mistake values to input data')
    print()


currect = 0
for test in range(testCount):
    spTest = sp.copy()




    if (noiseRate > 0):
        choices = [-1, 1]

        for i in range(sn):
            changedElements = []
            while True:
                randomNum = random.randint(0, sm-1)

                if randomNum not in changedElements:
                    spTest[i][randomNum] = random.choice(choices)

                changedElements.append(randomNum)

                if (len(changedElements) >= round(noiseRate * sm)):
                    break


    if (mistakeRate > 0):
        for i in range(sn):
            changedElements = []
            while True:
                randomNum = random.randint(0, sm-1)

                if randomNum not in changedElements:
                    spTest[i][randomNum] = 0

                changedElements.append(randomNum)

                if (len(changedElements) >= round(mistakeRate * sm)):
                    break



    for p in range(len(spTest)):
        if np.array_equal(np.sign(np.matmul(np.transpose(spTest[p]), w)), sp[p]):
            currect += 1

print('Result: ' + str(currect) + ' out of ' + str(sn*testCount) + ' succedded (' + str(round(currect / sn / testCount * 100)) + '%).')
