import torch

from arch.tokenization import BertTokenizer
import nltk.translate as nt

from xtec_data import tokenizer

useGPU = torch.cuda.is_available()
device = torch.device('cuda') if useGPU else torch.device('cpu')


import numpy as np







def bleuScore(captions, prediction, ngrams=[4]):
    if isinstance(captions, str):
        captions = [captions]

    scores = {}
    for ngram in ngrams:
        if ngram == 1:
            scores['b1'] = nt.bleu_score.sentence_bleu(captions, prediction, weights=(1, 0, 0, 0))
        elif ngram == 2:
            scores['b2'] = nt.bleu_score.sentence_bleu(captions, prediction, weights=(0, 1, 0, 0))
        elif ngram == 3:
            scores['b3'] = nt.bleu_score.sentence_bleu(captions, prediction, weights=(0, 0, 1, 0))
        elif ngram == 4:
            scores['b4'] = nt.bleu_score.sentence_bleu(captions, prediction, weights=(0, 0, 0, 1))

    return scores




def bulkBleuScore(captions, predictions, ngrams=[4]):
    scoreList = []
    scores = {}

    for ngram in ngrams:
        scores['b' + str(ngram)] = 0

    for i in range(len(captions)):
        bleu = bleuScore(captions[i], predictions[i], ngrams)

        for ngram in ngrams:
            scores['b' + str(ngram)] += bleu['b' + str(ngram)]

    for ngram in ngrams:
        scores['b' + str(ngram)] /= len(captions)

    return scores




def meteorScore(captions, prediction):
    if isinstance(captions, str):
        captions = [captions]

    return nt.meteor_score.meteor_score(captions, prediction)




def bulkMeteorScore(captions, predictions):
    totalScore = 0

    for i in range(len(captions)):
        totalScore += meteorScore(captions[i], predictions[i])

    return scoreList / len(captions)




