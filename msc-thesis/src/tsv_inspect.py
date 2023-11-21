import sys
import csv
import base64
import time
import pickle

import numpy as np


dataDirectoryName = 'data/mscoco_xtec/'

data = []

# for imageId in imageIds:

imageId = 'COCO_train2014_000000205790'

if imageId.find('COCO_train2014_') != -1:
    fileName = 'train2014'
elif imageId.find('COCO_val2014_') != -1:
    fileName = 'val2014'
else:
    fileName = 'test2015'

imageDataFile = open(dataDirectoryName + fileName + '/' + imageId + '.xtec', 'rb')
imageData = pickle.load(imageDataFile)
imageDataFile.close()

for key in ['img_h', 'img_w', 'num_boxes']:
    imageData[key] = int(imageData[key])

boxes = imageData['num_boxes']
decode_config = [
    ('objects_id', (boxes, ), np.int64),
    ('objects_conf', (boxes, ), np.float32),
    ('attrs_id', (boxes, ), np.int64),
    ('attrs_conf', (boxes, ), np.float32),
    ('boxes', (boxes, 4), np.float32),
    ('features', (boxes, -1), np.float32),
]
for key, shape, dtype in decode_config:
    imageData[key] = np.frombuffer(base64.b64decode(imageData[key]), dtype=dtype)
    imageData[key] = imageData[key].reshape(shape)
    imageData[key].setflags(write=False)

data.append(imageData)


print(imageData)
