import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import combinations

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



print('Testing...')


totalCurrect = []
for badElements in range(sm):
    currect = 0

    combs = combinations(range(sm), badElements)
    cnt = 0

    for comb in combs:
        cnt += 1

        for p in range(sn):
            spTest = sp.copy()

            for index in comb:
                spTest[p][index] = 0

            if np.array_equal(np.sign(np.matmul(np.transpose(spTest[p]), w)), sp[p]):
                currect += 1

    totalCurrect.append(currect / sn / cnt * 100)

    print('Result: Noisy elements: ' + str(badElements) + ' Correct: ' + str(currect) + ' out of ' + str(sn * cnt) + ' succedded (' + str(round(currect / sn / cnt * 100)) + '%).')



plt.plot(range(len(totalCurrect)), totalCurrect, label = "Accuracy")
plt.xlabel('Noised Elements')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Noise Level plot')
plt.legend()
plt.show()
