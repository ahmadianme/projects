import numpy as np
from sympy import *
from pyds import powerset
from pprint import pprint



def calcLambda(g):
    l = symbols('l')
    equation = 1

    for gItem in g:
        equation *= (1 + l * gItem)

    equation = Eq(equation - l - 1, 0)

    solutions = solve(equation, l)

    gSum = np.sum(g)

    if gSum == 1:
        return 0
    elif gSum < 1:
        for solution in solutions:
            if solution > 0:
                return solution
    elif gSum > 1:
        for solution in solutions:
            if solution < 0 and solution > -1:
                return solution

    return None




def powerSet(g):
    l = calcLambda(g)

    gAsMap = powerset(range(len(g)))
    gAs = dict.fromkeys(gAsMap, 0)

    for A in gAs:
        if len(A) == 1:
            gAs[A] = g[list(A)[0]]

        elif len(A) == 2:
            gVariables = list(A)
            gAs[A] = g[gVariables[0]] + g[gVariables[1]] + (l * g[gVariables[0]] * g[gVariables[1]])

        elif len(A) == 3 and len(A) != len(g):
            gVariables = list(A)
            gAs[A] =\
              g[gVariables[0]] + g[gVariables[1]] + g[gVariables[2]]\
              + (l * g[gVariables[0]] * g[gVariables[1]]) + (l * g[gVariables[0]] * g[gVariables[2]])\
              + (l * g[gVariables[1]] * g[gVariables[2]])\
              + (l**2 * g[gVariables[0]] * g[gVariables[1]] * g[gVariables[2]])

        elif len(A) == len(g):
            gAs[A] = 1

    return gAs



def sugeno(f, g):
    gAs = powerSet(g)

    fDict = dict(zip(range(len(f)), f))

    fDict = dict(sorted(fDict.items(), key=lambda x:x[1], reverse=False))

    minValues = []

    for key, fKey in enumerate(fDict.keys()):
        gA = gAs[frozenset(list(fDict.keys())[key:])]
        minValues.append(np.minimum(fDict[fKey], gA))

    return np.max(minValues)






def choquet(f, g):
    gAs = powerSet(g)

    fDict = dict(zip(range(len(f)), f))

    fDict = dict(sorted(fDict.items(), key=lambda x:x[1], reverse=False))

    sum = 0
    fBefore = 0

    for key, fKey in enumerate(fDict.keys()):
        gA = gAs[frozenset(list(fDict.keys())[key:])]
        sum += (fDict[fKey] - fBefore) * gA

        fBefore = fDict[fKey]

    return sum










print('2 Person {P1, P2} -----------------------------------------------------------------------------------------')

g = [0.3, 0.6]

fA = [1.0, 0.8]
fB = [0.5, 0.5]
fC = [0.3, 0.3]
fD = [0.3, 0.3]


sugenoA = sugeno(fA, g)
sugenoB = sugeno(fB, g)
sugenoC = sugeno(fC, g)
sugenoD = sugeno(fD, g)
#
choquetA = choquet(fA, g)
choquetB = choquet(fB, g)
choquetC = choquet(fC, g)
choquetD = choquet(fD, g)



l = calcLambda(g)
print('Lambda: ' + str(l))
print()
print()

gAs = powerSet(g)
print('g(A):')
pprint(gAs)
print()
print()



print('Sugeno A: ' + str(sugenoA))
print('Sugeno B: ' + str(sugenoB))
print('Sugeno C: ' + str(sugenoC))
print('Sugeno D: ' + str(sugenoD))
print()
print()

print('Choquet A: ' + str(choquetA))
print('Choquet B: ' + str(choquetB))
print('Choquet C: ' + str(choquetC))
print('Choquet D: ' + str(choquetD))
print()
print()








print('2 Person {P1, P3} -----------------------------------------------------------------------------------------')

g = [0.3, 0.7]

fA = [1.0, 0.1]
fB = [0.5, 0.3]
fC = [0.3, 0.2]
fD = [0.3, 0.8]


sugenoA = sugeno(fA, g)
sugenoB = sugeno(fB, g)
sugenoC = sugeno(fC, g)
sugenoD = sugeno(fD, g)
#
choquetA = choquet(fA, g)
choquetB = choquet(fB, g)
choquetC = choquet(fC, g)
choquetD = choquet(fD, g)



l = calcLambda(g)
print('Lambda: ' + str(l))
print()
print()

gAs = powerSet(g)
print('g(A):')
pprint(gAs)
print()
print()



print('Sugeno A: ' + str(sugenoA))
print('Sugeno B: ' + str(sugenoB))
print('Sugeno C: ' + str(sugenoC))
print('Sugeno D: ' + str(sugenoD))
print()
print()

print('Choquet A: ' + str(choquetA))
print('Choquet B: ' + str(choquetB))
print('Choquet C: ' + str(choquetC))
print('Choquet D: ' + str(choquetD))
print()
print()







print('2 Person {P1, P4} -----------------------------------------------------------------------------------------')

g = [0.3, 0.3]

fA = [1.0, 0.8]
fB = [0.5, 0.7]
fC = [0.3, 0.4]
fD = [0.3, 0.8]


sugenoA = sugeno(fA, g)
sugenoB = sugeno(fB, g)
sugenoC = sugeno(fC, g)
sugenoD = sugeno(fD, g)
#
choquetA = choquet(fA, g)
choquetB = choquet(fB, g)
choquetC = choquet(fC, g)
choquetD = choquet(fD, g)



l = calcLambda(g)
print('Lambda: ' + str(l))
print()
print()

gAs = powerSet(g)
print('g(A):')
pprint(gAs)
print()
print()



print('Sugeno A: ' + str(sugenoA))
print('Sugeno B: ' + str(sugenoB))
print('Sugeno C: ' + str(sugenoC))
print('Sugeno D: ' + str(sugenoD))
print()
print()

print('Choquet A: ' + str(choquetA))
print('Choquet B: ' + str(choquetB))
print('Choquet C: ' + str(choquetC))
print('Choquet D: ' + str(choquetD))
print()
print()







print('2 Person {P2, P3} -----------------------------------------------------------------------------------------')

g = [0.6, 0.7]

fA = [0.8, 0.1]
fB = [0.5, 0.3]
fC = [0.3, 0.2]
fD = [0.3, 0.8]


sugenoA = sugeno(fA, g)
sugenoB = sugeno(fB, g)
sugenoC = sugeno(fC, g)
sugenoD = sugeno(fD, g)
#
choquetA = choquet(fA, g)
choquetB = choquet(fB, g)
choquetC = choquet(fC, g)
choquetD = choquet(fD, g)



l = calcLambda(g)
print('Lambda: ' + str(l))
print()
print()

gAs = powerSet(g)
print('g(A):')
pprint(gAs)
print()
print()



print('Sugeno A: ' + str(sugenoA))
print('Sugeno B: ' + str(sugenoB))
print('Sugeno C: ' + str(sugenoC))
print('Sugeno D: ' + str(sugenoD))
print()
print()

print('Choquet A: ' + str(choquetA))
print('Choquet B: ' + str(choquetB))
print('Choquet C: ' + str(choquetC))
print('Choquet D: ' + str(choquetD))
print()
print()






print('2 Person {P2, P4} -----------------------------------------------------------------------------------------')

g = [0.6, 0.3]

fA = [0.8, 0.8]
fB = [0.5, 0.7]
fC = [0.3, 0.4]
fD = [0.3, 0.8]


sugenoA = sugeno(fA, g)
sugenoB = sugeno(fB, g)
sugenoC = sugeno(fC, g)
sugenoD = sugeno(fD, g)
#
choquetA = choquet(fA, g)
choquetB = choquet(fB, g)
choquetC = choquet(fC, g)
choquetD = choquet(fD, g)



l = calcLambda(g)
print('Lambda: ' + str(l))
print()
print()

gAs = powerSet(g)
print('g(A):')
pprint(gAs)
print()
print()



print('Sugeno A: ' + str(sugenoA))
print('Sugeno B: ' + str(sugenoB))
print('Sugeno C: ' + str(sugenoC))
print('Sugeno D: ' + str(sugenoD))
print()
print()

print('Choquet A: ' + str(choquetA))
print('Choquet B: ' + str(choquetB))
print('Choquet C: ' + str(choquetC))
print('Choquet D: ' + str(choquetD))
print()
print()







print('2 Person {P3, P4} -----------------------------------------------------------------------------------------')

g = [0.7, 0.3]

fA = [0.1, 0.8]
fB = [0.3, 0.7]
fC = [0.2, 0.4]
fD = [0.8, 0.8]


sugenoA = sugeno(fA, g)
sugenoB = sugeno(fB, g)
sugenoC = sugeno(fC, g)
sugenoD = sugeno(fD, g)
#
choquetA = choquet(fA, g)
choquetB = choquet(fB, g)
choquetC = choquet(fC, g)
choquetD = choquet(fD, g)



l = calcLambda(g)
print('Lambda: ' + str(l))
print()
print()

gAs = powerSet(g)
print('g(A):')
pprint(gAs)
print()
print()



print('Sugeno A: ' + str(sugenoA))
print('Sugeno B: ' + str(sugenoB))
print('Sugeno C: ' + str(sugenoC))
print('Sugeno D: ' + str(sugenoD))
print()
print()

print('Choquet A: ' + str(choquetA))
print('Choquet B: ' + str(choquetB))
print('Choquet C: ' + str(choquetC))
print('Choquet D: ' + str(choquetD))
print()
print()









print('3 Person {P1, P2, P3} -----------------------------------------------------------------------------------------')

g = [0.3, 0.6, 0.7]

fA = [1.0, 0.8, 0.1]
fB = [0.5, 0.5, 0.3]
fC = [0.3, 0.3, 0.2]
fD = [0.3, 0.3, 0.8]


sugenoA = sugeno(fA, g)
sugenoB = sugeno(fB, g)
sugenoC = sugeno(fC, g)
sugenoD = sugeno(fD, g)
#
choquetA = choquet(fA, g)
choquetB = choquet(fB, g)
choquetC = choquet(fC, g)
choquetD = choquet(fD, g)



l = calcLambda(g)
print('Lambda: ' + str(l))
print()
print()

gAs = powerSet(g)
print('g(A):')
pprint(gAs)
print()
print()



print('Sugeno A: ' + str(sugenoA))
print('Sugeno B: ' + str(sugenoB))
print('Sugeno C: ' + str(sugenoC))
print('Sugeno D: ' + str(sugenoD))
print()
print()

print('Choquet A: ' + str(choquetA))
print('Choquet B: ' + str(choquetB))
print('Choquet C: ' + str(choquetC))
print('Choquet D: ' + str(choquetD))
print()
print()






print('3 Person {P1, P2, P4} -----------------------------------------------------------------------------------------')

g = [0.3, 0.6, 0.3]

fA = [1.0, 0.8, 0.8]
fB = [0.5, 0.5, 0.7]
fC = [0.3, 0.3, 0.4]
fD = [0.3, 0.3, 0.8]


sugenoA = sugeno(fA, g)
sugenoB = sugeno(fB, g)
sugenoC = sugeno(fC, g)
sugenoD = sugeno(fD, g)
#
choquetA = choquet(fA, g)
choquetB = choquet(fB, g)
choquetC = choquet(fC, g)
choquetD = choquet(fD, g)



l = calcLambda(g)
print('Lambda: ' + str(l))
print()
print()

gAs = powerSet(g)
print('g(A):')
pprint(gAs)
print()
print()



print('Sugeno A: ' + str(sugenoA))
print('Sugeno B: ' + str(sugenoB))
print('Sugeno C: ' + str(sugenoC))
print('Sugeno D: ' + str(sugenoD))
print()
print()

print('Choquet A: ' + str(choquetA))
print('Choquet B: ' + str(choquetB))
print('Choquet C: ' + str(choquetC))
print('Choquet D: ' + str(choquetD))
print()
print()







print('3 Person {P1, P3, P4} -----------------------------------------------------------------------------------------')

g = [0.3, 0.7, 0.3]

fA = [1.0, 0.1, 0.8]
fB = [0.5, 0.3, 0.7]
fC = [0.3, 0.2, 0.4]
fD = [0.3, 0.8, 0.8]


sugenoA = sugeno(fA, g)
sugenoB = sugeno(fB, g)
sugenoC = sugeno(fC, g)
sugenoD = sugeno(fD, g)
#
choquetA = choquet(fA, g)
choquetB = choquet(fB, g)
choquetC = choquet(fC, g)
choquetD = choquet(fD, g)



l = calcLambda(g)
print('Lambda: ' + str(l))
print()
print()

gAs = powerSet(g)
print('g(A):')
pprint(gAs)
print()
print()



print('Sugeno A: ' + str(sugenoA))
print('Sugeno B: ' + str(sugenoB))
print('Sugeno C: ' + str(sugenoC))
print('Sugeno D: ' + str(sugenoD))
print()
print()

print('Choquet A: ' + str(choquetA))
print('Choquet B: ' + str(choquetB))
print('Choquet C: ' + str(choquetC))
print('Choquet D: ' + str(choquetD))
print()
print()







print('3 Person {P2, P3, P4} -----------------------------------------------------------------------------------------')

g = [0.6, 0.7, 0.3]

fA = [0.8, 0.1, 0.8]
fB = [0.5, 0.3, 0.7]
fC = [0.3, 0.2, 0.4]
fD = [0.3, 0.8, 0.8]


sugenoA = sugeno(fA, g)
sugenoB = sugeno(fB, g)
sugenoC = sugeno(fC, g)
sugenoD = sugeno(fD, g)
#
choquetA = choquet(fA, g)
choquetB = choquet(fB, g)
choquetC = choquet(fC, g)
choquetD = choquet(fD, g)



l = calcLambda(g)
print('Lambda: ' + str(l))
print()
print()

gAs = powerSet(g)
print('g(A):')
pprint(gAs)
print()
print()



print('Sugeno A: ' + str(sugenoA))
print('Sugeno B: ' + str(sugenoB))
print('Sugeno C: ' + str(sugenoC))
print('Sugeno D: ' + str(sugenoD))
print()
print()

print('Choquet A: ' + str(choquetA))
print('Choquet B: ' + str(choquetB))
print('Choquet C: ' + str(choquetC))
print('Choquet D: ' + str(choquetD))
print()
print()







print('4 Person {P1, P2, P3, P4} -----------------------------------------------------------------------------------------')

g = [0.3, 0.6, 0.7, 0.3]
# g = [0.3/1.9, 0.6/1.9, 0.7/1.9, 0.3/1.9] # weighted arithmetic mean

fA = [1.0, 0.8, 0.1, 0.8]
fB = [0.5, 0.5, 0.3, 0.7]
fC = [0.3, 0.3, 0.2, 0.4]
fD = [0.3, 0.3, 0.8, 0.8]


sugenoA = sugeno(fA, g)
sugenoB = sugeno(fB, g)
sugenoC = sugeno(fC, g)
sugenoD = sugeno(fD, g)
#
choquetA = choquet(fA, g)
choquetB = choquet(fB, g)
choquetC = choquet(fC, g)
choquetD = choquet(fD, g)



l = calcLambda(g)
print('Lambda: ' + str(l))
print()
print()

gAs = powerSet(g)
print('g(A):')
pprint(gAs)
print()
print()



print('Sugeno A: ' + str(sugenoA))
print('Sugeno B: ' + str(sugenoB))
print('Sugeno C: ' + str(sugenoC))
print('Sugeno D: ' + str(sugenoD))
print()
print()

print('Choquet A: ' + str(choquetA))
print('Choquet B: ' + str(choquetB))
print('Choquet C: ' + str(choquetC))
print('Choquet D: ' + str(choquetD))
print()
print()
