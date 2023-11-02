import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier






def calculateEntropy(dataY):
    return np.sum(-dataY.value_counts()/dataY.shape[0]*np.log2(dataY.value_counts()/dataY.shape[0]+1e-9))

def findBestDecision(x, dataY):
    splitValue = []
    infoGain = []

    numVar = True if x.dtypes != 'O' else False

    if numVar:
        options = x.sort_values().unique()[1:]
    else:
        xTmp = x.unique()

        optionsTmp = []
        for L in range(0, len(xTmp)+1):
            for subset in itertools.combinations(xTmp, L):
                subset = list(subset)
                optionsTmp.append(subset)

        options = optionsTmp[1:-1]

    for value in options:
        mask = x < value if numVar else x.isin(value)

        a = sum(mask)
        b = mask.shape[0] - a

        if(a == 0 or b == 0):
            infoGainValue = 0

        else:
            infoGainValue = calculateEntropy(dataY)-a/(a+b)*calculateEntropy(dataY[mask])-b/(a+b)*calculateEntropy(dataY[-mask])

        infoGain.append(infoGainValue)
        splitValue.append(value)

    if len(infoGain) == 0:
        return(None, None, None, False)

    else:
        bestInfoGain = max(infoGain)
        return(bestInfoGain, splitValue[infoGain.index(bestInfoGain)], numVar, True)

def predict(data, tf):
    if tf:
        predictions = data.value_counts().idxmax()
    else:
        predictions = data.mean()

    return predictions

def train(data, dataY, tf, depth=None, minSamplesSplit=2, minInfoGain=1e-20, c=0):
    if depth == None:
        depthCondition = True

    else:
        if c < depth:
            depthCondition = True
        else:
            depthCondition = False

    if minSamplesSplit == None:
        sampleCondition = True

    else:
        if data.shape[0] > minSamplesSplit:
            sampleCondition = True
        else:
            sampleCondition = False

    if depthCondition & sampleCondition:
        masks = data.drop(dataY, axis= 1).apply(findBestDecision, dataY = data[dataY])
        masks.reset_index(inplace=True)
        masks.drop(["index"], axis=1, inplace=True)
        masks = masks.loc[:,masks.loc[3,:]]

        variable = max(masks)
        value = masks[variable][1]
        infoGain = masks[variable][0]
        variableType = masks[variable][2]

        if infoGain is not None and infoGain >= minInfoGain:
            c += 1

            if variableType:
                left = data[data[variable] < value]
                right = data[(data[variable] < value) == False]

            else:
                left = data[data[variable].isin(value)]
                right = data[(data[variable].isin(value)) == False]

            st = "<=" if variableType else "in"
            dec =   "{} {}  {}".format(variable, st, value)
            subtree = {dec: []}

            yes = train(left, dataY, tf, depth, minSamplesSplit, minInfoGain, c)
            no = train(right, dataY, tf, depth, minSamplesSplit, minInfoGain, c)

            if yes == no:
                subtree = yes
            else:
                subtree[dec].append(yes)
                subtree[dec].append(no)
        else:
            predictions = predict(data[dataY], tf)
            return predictions
    else:
        predictions = predict(data[dataY], tf)
        return predictions

    return subtree






dataframe = pd.DataFrame(data={
    'Alt':
        ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'No', 'Yes'],
    'Bar':
        ['No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'Fri':
        ['No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'Hun':
        ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'Pat':
        ['Some', 'Full', 'Some', 'Full', 'Full', 'Some', 'None', 'Some', 'Full', 'Full', 'None', 'Full'],
    'Price':
        ['$$$', '$', '$', '$', '$$$', '$$', '$', '$$', '$', '$$$', '$', '$'],
    'Rain':
        ['No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No'],
    'Res':
        ['Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No'],
    'Type':
        ['French', 'Thai', 'Burger', 'Thai', 'French', 'Italian', 'Burger', 'Thai', 'Burger', 'Italian', 'Thai', 'Burger'],
    'Est':
        ['0-10', '30-60', '0-10', '10-30', '>60', '0-10', '0-10', '0-10', '>60', '10-30', '0-10', '30-60'],
    'WillWait':
        ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes'],
})


finalTree = train(dataframe, 'WillWait', True, depth=5)
print(finalTree)









classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5)
dataX = dataframe.iloc[:, :-1].apply(lambda x: pd.factorize(x)[0])
dataY = dataframe.iloc[:, -1].replace({'No': 0, 'Yes': 1})
classifier = classifier.fit(dataX, dataY)
tree.plot_tree(classifier, feature_names=dataframe.columns[:-1],  class_names=['No', 'Yes'], filled=True)
plt.show()
