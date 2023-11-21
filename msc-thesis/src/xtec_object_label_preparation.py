# coding=utf-8

import json
import os
import pickle
import string
import numpy as np
from collections import OrderedDict

MAX_CAPTION_LENGTH = 20




labels = {}
maxLength = 0

rawData = json.load(open("data/xtec/raw/labels/1imageLabels-eval1.json"))

for rawDatum in rawData:
    labels[rawDatum['image_id'][0]] = ' '.join(list(set(rawDatum['labels'][0].lower().strip().split())))
    obArray = labels[rawDatum['image_id'][0]].split()
    if maxLength < len(obArray):
        maxLength = len(obArray)
        



rawData = json.load(open("data/xtec/raw/labels/1imageLabels-eval2.json"))

for rawDatum in rawData:
    labels[rawDatum['image_id'][0]] = ' '.join(list(set(rawDatum['labels'][0].lower().strip().split())))
    obArray = labels[rawDatum['image_id'][0]].split()
    if maxLength < len(obArray):
        maxLength = len(obArray)
        



rawData = json.load(open("data/xtec/raw/labels/1imageLabels-eval3.json"))

for rawDatum in rawData:
    labels[rawDatum['image_id'][0]] = ' '.join(list(set(rawDatum['labels'][0].lower().strip().split())))
    obArray = labels[rawDatum['image_id'][0]].split()
    if maxLength < len(obArray):
        maxLength = len(obArray)
        



rawData = json.load(open("data/xtec/raw/labels/1imageLabels-eval.json"))

for rawDatum in rawData:
    labels[rawDatum['image_id'][0]] = ' '.join(list(set(rawDatum['labels'][0].lower().strip().split())))
    obArray = labels[rawDatum['image_id'][0]].split()
    if maxLength < len(obArray):
        maxLength = len(obArray)
        



rawData = json.load(open("data/xtec/raw/labels/imageLabels-eval1.json"))

for rawDatum in rawData:
    labels[rawDatum['image_id'][0]] = ' '.join(list(set(rawDatum['labels'][0].lower().strip().split())))
    obArray = labels[rawDatum['image_id'][0]].split()
    if maxLength < len(obArray):
        maxLength = len(obArray)
        



rawData = json.load(open("data/xtec/raw/labels/imageLabels-eval2.json"))

for rawDatum in rawData:
    labels[rawDatum['image_id'][0]] = ' '.join(list(set(rawDatum['labels'][0].lower().strip().split())))
    obArray = labels[rawDatum['image_id'][0]].split()
    if maxLength < len(obArray):
        maxLength = len(obArray)
        



rawData = json.load(open("data/xtec/raw/labels/imageLabels-eval3.json"))

for rawDatum in rawData:
    labels[rawDatum['image_id'][0]] = ' '.join(list(set(rawDatum['labels'][0].lower().strip().split())))
    obArray = labels[rawDatum['image_id'][0]].split()
    if maxLength < len(obArray):
        maxLength = len(obArray)
        



rawData = json.load(open("data/xtec/raw/labels/imageLabels-eval.json"))

for rawDatum in rawData:
    labels[rawDatum['image_id'][0]] = ' '.join(list(set(rawDatum['labels'][0].lower().strip().split())))
    obArray = labels[rawDatum['image_id'][0]].split()
    if maxLength < len(obArray):
        maxLength = len(obArray)
        



rawData = json.load(open("data/xtec/raw/labels/imageLabels.json"))

for rawDatum in rawData:
    labels[rawDatum['image_id'][0]] = ' '.join(list(set(rawDatum['labels'][0].lower().strip().split())))
    obArray = labels[rawDatum['image_id'][0]].split()
    if maxLength < len(obArray):
        maxLength = len(obArray)
        



print('Total Count:', str(len(labels)))
print('Max Length:', str(maxLength))


jsonFile = open("data/xtec/object_classes.json", "w")
jsonFile.write(json.dumps(labels))
jsonFile.close()
print('File has been saved: object_classes.json')



exit()
