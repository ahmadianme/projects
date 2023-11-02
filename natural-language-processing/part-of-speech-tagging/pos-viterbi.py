import nltk
import numpy as np
import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm







smoothingRate = 0.000001

print('Smoothing Rate: ' + str(smoothingRate))





nltk.download('treebank')
nltk.download('universal_tagset')





NLTKDataset = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))



trainData, testData =train_test_split(NLTKDataset,train_size=0.80, test_size=0.20, random_state = 101)
# trainData,validation_set =train_test_split(trainData,train_size=0.85, test_size=0.15, random_state = 101)






allTagedWords = [ word for sentence in NLTKDataset for word in sentence ]
trainTaggedWords = [ word for sentence in trainData for word in sentence ]
testTaggedWords = [ word for sentence in testData for word in sentence ]


tags = {tag for word, tag in allTagedWords}





emmisionMatrix = {}

def calculateEmmisionValue(word, tag, trainItems = trainTaggedWords):
    wordTag = str(word) + '-!-' + str(tag)

    if wordTag not in emmisionMatrix:
        tagList = [pair for pair in trainItems if pair[1]==tag]
        tagCount = len(tagList)
        wordTagProbabilityList = [pair[0] for pair in tagList if pair[0]==word]
        wordTagProbabilityCount = len(wordTagProbabilityList)
        emmisionMatrix[wordTag] = wordTagProbabilityCount / tagCount

        if emmisionMatrix[wordTag] == 0:
            emmisionMatrix[wordTag] = smoothingRate


    return emmisionMatrix[wordTag]








def calculateTransitionValue(t2, t1, trainItems = trainTaggedWords):
    tags = [pair[1] for pair in trainItems]
    tag1Count = len([t for t in tags if t==t1])
    tag1tag2Count = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            tag1tag2Count += 1

    return tag1tag2Count / tag1Count







tagsMatrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)):
        trans = calculateTransitionValue(t2, t1)

        tagsMatrix[i, j] = trans








tagsDataFrame = pd.DataFrame(tagsMatrix, columns = list(tags), index=list(tags))



def Viterbi(words, trainItems = trainTaggedWords):
    state = []
    tagList = list(set([pair[1] for pair in trainItems]))

    with tqdm(total=len(words)) as pbar:
        for key, word in enumerate(words):
            probabilities = []
            for tag in tagList:
                if key == 0:
                    transitionProbability = tagsDataFrame.loc['.', tag]
                else:
                    transitionProbability = tagsDataFrame.loc[state[-1], tag]

                emissionProbability = calculateEmmisionValue(words[key], tag)
                stateProbability = emissionProbability * transitionProbability
                probabilities.append(stateProbability)

            maxProbability = max(probabilities)
            state_max = tagList[probabilities.index(maxProbability)]
            state.append(state_max)

            pbar.update(1)

    return list(zip(words, state))





testLen = round(len(testData) * 0.01)
testSetDev = testData[:testLen]

testTaggedWords = [tup for sent in testSetDev for tup in sent]
testWordsUntaged = [tup[0] for sent in testSetDev for tup in sent]



start = time.time()
predictedWords = Viterbi(testWordsUntaged)
end = time.time()
difference = end-start

print("Time taken in seconds: ", difference)

check = [i for i, j in zip(testTaggedWords, predictedWords) if i == j]
incorrectClassifiedWords = [(i, j) for i, j in zip(testTaggedWords, predictedWords) if i != j]

predictions = []
correctLabels = []

for i in range(len(predictedWords)):
    predictions.append(predictedWords[i][1])
    correctLabels.append(testTaggedWords[i][1])

microPrecision = precision_score(correctLabels, predictions, average='micro')
microRecall = recall_score(correctLabels, predictions, average='micro')
microF1 = f1_score(correctLabels, predictions, average='micro')

macroPrecision = precision_score(correctLabels, predictions, average='macro')
macroRecall = recall_score(correctLabels, predictions, average='macro')
macroF1 = f1_score(correctLabels, predictions, average='macro')



print('Micro Averaged Precision: ' + str(microPrecision))
print('Micro Averaged Recall: ' + str(microRecall))
print('Micro Averaged F1: ' + str(microF1))
print()
print('Macro Averaged Precision: ' + str(macroPrecision))
print('Macro Averaged Recall: ' + str(macroRecall))
print('Macro Averaged F1: ' + str(macroF1))



print()
accuracy = len(check)/len(predictedWords)
print('Viterbi Algorithm Accuracy: ',accuracy*100)
