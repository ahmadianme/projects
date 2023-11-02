import re
from operator import itemgetter
from typing import Dict, Tuple, List



# hyperparameters
iterationCount = 8



def preProcessCorpus(corpus):
    words = corpus.split()

    for i in range(len(words)):
        words[i] = " ".join(words[i])
        words[i] += ' _'

    newCorpus = {}
    for word in words:
        if word not in newCorpus:
            newCorpus[word] = 1
        else:
            newCorpus[word] += 1

    return newCorpus




def trainBPE(corpus):
    print()

    bpeMerges = {}

    for i in range(iterationCount):
        print('Iteration ' + str(i + 1))
        frequencies = {}

        for word, frequency in corpus.items():
            entries = word.split()

            for i in range(len(entries) - 1):
                pair = (entries[i], entries[i + 1])
                frequencyTmp = frequencies.get(pair, 0)
                frequencies[pair] = frequencyTmp + frequency

        if not frequencies:
            break

        bestPair = max(frequencies, key=frequencies.get)
        bpeMerges[bestPair] = i

        print('Corpus: ' + str(corpus))
        print('Best pair:' + str(bestPair))

        for frequency in frequencies:
            count, pair = frequency
            print(str(frequencies[frequency]) + '   ' + frequency[0] + frequency[1])

        print()

        corpusTemp = {}

        pattern = re.escape(' '.join(bestPair))
        replacement = ''.join(bestPair)

        for word in corpus:
            newWord = re.sub(pattern, replacement, word)
            corpusTemp[newWord] = corpus[word]

        corpus = corpusTemp

    print('Merges: ' + str(bpeMerges))
    print()

    return bpeMerges




def tokenize(testWord: str, bpeMerges: Dict[Tuple[str, str], int]) -> List[str]:
    if len(testWord) == 1:
        return testWord

    word = list(testWord)
    word.append('_')

    while True:
        pairs = set()

        previousCharacter = word[0]

        for character in word[1:]:
            pairs.add((previousCharacter, character))
            previousCharacter = character

        bpe_codes_pairs = [(pair, bpeMerges[pair]) for pair in pairs if pair in bpeMerges]
        if not bpe_codes_pairs:
            break

        pair_to_merge = min(bpe_codes_pairs, key=itemgetter(1))[0]

        first, second = pair_to_merge
        newWord = []

        i = 0
        wordLen = len(word)

        while i < wordLen:
            try:
                j = word.index(first, i)
                newWord.extend(word[i:j])
                i = j
            except ValueError:
                newWord.extend(word[i:])
                break

            if i < len(word) - 1 and word[i + 1] == second:
                newWord.append(first + second)
                i += 2
            else:
                newWord.append(first)
                i += 1

        word = newWord

    return word





# training phase
print('Training BPE on the corpus...')

corpus = 'low low low low low lower lower widest widest widest newest newest newest newest newest'
corpus = preProcessCorpus(corpus)

bpeMerges = trainBPE(corpus)






# test phase
print('Testing...')

testWord = 'lowest'
print('Test word: ' + testWord)

encoded = tokenize(testWord, bpeMerges)
print('Tokenized word: ' + str(encoded))
