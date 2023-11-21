# coding=utf-8

import json
import os
import pickle
import string
import numpy as np
from collections import OrderedDict

MAX_CAPTION_LENGTH = 20






test2015Data = json.load(open("data/xtec/raw/image_info_test2015.json"))
test2015Data = [dict({'img_id': 'COCO_test2015_' + str(img['id']).zfill(12) ,'caption': ''}) for img in test2015Data['images']]
print('Total Test2015 entry count: ' + str(len(test2015Data)))



test2015DevData = json.load(open("data/xtec/raw/image_info_test-dev2015.json"))
test2015DevData = [dict({'img_id': 'COCO_test2015_' + str(img['id']).zfill(12) ,'caption': ''}) for img in test2015DevData['images']]
print('Total Test2015-Dev entry count: ' + str(len(test2015DevData)))












stringTrans = str.maketrans('', '', string.punctuation)

# load and clean train data
trainData = json.load(open("data/xtec/raw/captions_train2014.json"))
trainData = trainData['annotations']

print('Total Train entry count: ' + str(len(trainData)))
print('Train Raw Data Sample:')
print (*trainData[:10], sep='\n')
print()

trainData = list(annotation for annotation in trainData if len(annotation['caption'].split()) <= MAX_CAPTION_LENGTH - 2)

for aIndex, a in enumerate(trainData):
    caption = list('<P>' for _ in range(MAX_CAPTION_LENGTH))
    # cap = a['caption'].lower().translate(stringTrans).strip().split()
    cap = a['caption'].lower().translate(stringTrans).strip() + ' .'
    # cl = len(cap) + 1

    # for i, word in enumerate(['<S>'] + cap + ['<E>']):
    #     caption[i] = word

    # ca = caption
    # caption = ' '.join(caption)


    # trainData[aIndex] = dict({'img_id': 'COCO_train2014_' + str(a['image_id']).zfill(12), 'caption_id': a['id'] ,'caption': caption, 'ca': ca, 'cl': cl})
    trainData[aIndex] = dict({'img_id': 'COCO_train2014_' + str(a['image_id']).zfill(12), 'caption_id': a['id'] ,'caption': cap})

# trainData = list(dict({'img_id': 'COCO_train2014_' + str(a['image_id']).zfill(12), 'caption_id': a['id'] ,'caption': a['caption'].lower().translate(stringTrans).strip()}) for a in trainData)

print('Train Prosseced Data Sample:')
print (*trainData[:10], sep='\n')
print()



# load and train validation data
validationData = json.load(open("data/xtec/raw/captions_val2014.json"))
validationData = validationData['annotations']

print('Total Validation entry count: ' + str(len(validationData)))
print('Validation Raw Data Sample:')
print (*validationData[:10], sep='\n')
print()

validationData = list(annotation for annotation in validationData if len(annotation['caption'].split()) <= MAX_CAPTION_LENGTH - 2)

for aIndex, a in enumerate(validationData):
    caption = list('<P>' for _ in range(MAX_CAPTION_LENGTH))
    # cap = a['caption'].lower().translate(stringTrans).strip().split()
    cap = a['caption'].lower().translate(stringTrans).strip() + " ."
    # cl = len(cap) + 1

    # for i, word in enumerate(['<S>'] + cap + ['<E>']):
    #     caption[i] = word

    # ca = caption
    # caption = ' '.join(caption)


    # validationData[aIndex] = dict({'img_id': 'COCO_val2014_' + str(a['image_id']).zfill(12), 'caption_id': a['id'] ,'caption': caption, 'ca': ca, 'cl': cl})
    validationData[aIndex] = dict({'img_id': 'COCO_val2014_' + str(a['image_id']).zfill(12), 'caption_id': a['id'] ,'caption': cap})

# validationData = list(dict({'img_id': 'COCO_val2014_' + str(a['image_id']).zfill(12), 'caption_id': a['id'] ,'caption': a['caption'].lower().translate(stringTrans).strip()}) for a in validationData)

print('Validation Prosseced Data Sample:')
print (*validationData[:10], sep='\n')
print()




# minival
vqaMinivalData = json.load(open("data/vqa/minival.json"))
vqaMinivalImgIds = list(data['img_id'] for data in vqaMinivalData)
minivalData = list(data for data in validationData if data['img_id'] in vqaMinivalImgIds)

print('Total minival entry count: ' + str(len(minivalData)))
print()




# create the dictionary
dictionarySorted = OrderedDict({})
# [dictionarySorted.append(word) for annotation in trainData for word in annotation['caption'].split() if word not in dictionarySorted]
{(dictionarySorted.update({word: dictionarySorted[word] + 1}) if word in dictionarySorted else dictionarySorted.update({word: 1})) for annotation in trainData for word in annotation['caption'].split()}
{(dictionarySorted.update({word: dictionarySorted[word] + 1}) if word in dictionarySorted else dictionarySorted.update({word: 1})) for annotation in validationData for word in annotation['caption'].split()}
dictionarySorted = dict(sorted(dictionarySorted.items(), key=lambda item: item[1], reverse=True))
# dictionarySorted.pop('<P>')
# dictionarySorted.pop('<S>')
# dictionarySorted.pop('<E>')

dictionary = ['<P>', '<S>', '<E>'] + list(dictionarySorted.keys())

print('Total Dictionary entry count: ' + str(len(dictionary)))
print('Dictionary Sample:')
print(dictionary[:30])
print()





# save proccessed train, validation and dictionary data to files
jsonFile = open("data/xtec/train.json", "w")
jsonFile.write(json.dumps(trainData))
jsonFile.close()
print('File has been saved: train.json')

jsonFile = open("data/xtec/val2014.json", "w")
jsonFile.write(json.dumps(validationData))
jsonFile.close()
print('File has been saved: val2014.json')

jsonFile = open("data/xtec/minival.json", "w")
jsonFile.write(json.dumps(minivalData))
jsonFile.close()
print('File has been saved: minival.json')

jsonFile = open("data/xtec/test2015.json", "w")
jsonFile.write(json.dumps(test2015Data))
jsonFile.close()
print('File has been saved: test2015.json')

jsonFile = open("data/xtec/test2015dev.json", "w")
jsonFile.write(json.dumps(test2015DevData))
jsonFile.close()
print('File has been saved: test2015dev.json')

jsonFile = open("data/xtec/trainval_label2word.json", "w")
jsonFile.write(json.dumps(dictionary))
jsonFile.close()
jsonFile = open("data/xtec/trainval_word2label.json", "w")
jsonFile.write(json.dumps(dict(zip(dictionary, range(len(dictionary))))))
jsonFile.close()
print('Files has been saved: trainval_word2label.json and trainval_label2word.json')

jsonFile = open("data/xtec/dictionary_word_count.json", "w")
jsonFile.write(json.dumps(dictionarySorted))
jsonFile.close()
print('Files has been saved: dictionary_word_count.json')
