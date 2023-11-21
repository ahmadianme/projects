import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from arguments import args
from utils import load_obj_tsv, load_obj_xtec
from arch.tokenization import BertTokenizer

useGPU = torch.cuda.is_available()
device = torch.device('cuda') if useGPU else torch.device('cpu')

TINY_IMG_NUM = 400
FAST_IMG_NUM = 5000

MAX_CAPTION_LENGTH = 20

XTEC_DATA_ROOT = 'data/xtec/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}





tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=True
)

clsTokenId = tokenizer.vocab["[CLS]"]
sepTokenId = tokenizer.vocab["[SEP]"]
maskTokenId = tokenizer.vocab["[MASK]"]
dotTokenId = tokenizer.vocab["."]



def load_image_features(imageIdList):
    imageData = load_obj_xtec(imageIdList)

    allFeats = []
    allBoxes = []

    for i, img_info in enumerate(imageData):
        # img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        allFeats.append(feats)
        allBoxes.append(boxes)

    allFeats = torch.tensor(allFeats)
    allBoxes = torch.tensor(allBoxes)

    return allFeats, allBoxes






def prepareLanguageFeatures(captions):
    features = []

    for caption in captions:
        tokens = tokenizer.tokenize(caption.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > args.maxCaptionLength - 2:
            tokens = tokens[:(args.maxCaptionLength - 2)]

        # concatenate lm labels and account for CLS, SEP, SEP
        input_ids = tokenizer.convert_tokens_to_ids(tokens)


        # Zero-pad up to the sequence length.
        while len(input_ids) < args.maxCaptionLength:
            input_ids.append(0)

        assert len(input_ids) == args.maxCaptionLength

        features.append(input_ids)

    features = torch.tensor(features, dtype=torch.long).to(device)

    return features





def prepareLanguageFeaturesStep(input_ids, wordPosition, lengths, objectClasses=None):
    validLenghts = torch.full((1, input_ids.size(0)), wordPosition).to(device)
    validLenghts = torch.min(validLenghts, lengths).flatten().cpu().numpy()

    if args.maxObjectClassLength > 0:
        objectClassesLength = torch.count_nonzero(objectClasses, dim=1).to(device)
        inputLength = input_ids.size(1) + args.maxObjectClassLength
    else:
        inputLength = input_ids.size(1)


    input_ids_tmp = torch.zeros(input_ids.size(0), inputLength, dtype=torch.long).to(device)
    lm_label_ids = torch.neg(torch.ones(input_ids.size(0), inputLength, dtype=torch.long)).to(device)
    input_mask = torch.ones(input_ids.size(0), inputLength, dtype=torch.long).to(device)
    segment_ids = torch.zeros(input_ids.size(0), inputLength, dtype=torch.long).to(device)

    input_ids_tmp[:, 0] = clsTokenId


    for i in range(input_ids.size(0)):
        if wordPosition <= validLenghts[i]:
            input_ids_tmp[i, 1:validLenghts[i]+1] = input_ids[i, :validLenghts[i]]
            input_ids_tmp[i, validLenghts[i]+1] = maskTokenId
            input_ids_tmp[i, validLenghts[i]+2] = sepTokenId
            lm_label_ids[i, validLenghts[i]+1] = input_ids[i, validLenghts[i]]
        else:
            input_ids_tmp[i, 1:validLenghts[i]+2] = input_ids[i, :validLenghts[i]+1]
            input_ids_tmp[i, validLenghts[i]+2] = sepTokenId

        if args.maxObjectClassLength > 0:
            input_ids_tmp[i, validLenghts[i]+3:objectClassesLength[i]+validLenghts[i]+3] = objectClasses[i, 0:objectClassesLength[i]]
            input_mask[i, validLenghts[i]+3+objectClassesLength[i]:] = 0
            # segment_ids[i, validLenghts[i]+3:objectClassesLength[i]+validLenghts[i]+3] = 1
            segment_ids[i, validLenghts[i]+3:] = 1
        else:
            input_mask[i, validLenghts[i]+3:] = 0





    # print()
    # print('inputLength ' + str(inputLength) + ' =======================================================')
    # print('input_ids')
    # print(input_ids)
    # print()
    # print('objectClasses')
    # print(objectClasses)
    # print()
    # # print('objectClasses[i, 0:objectClassesLength[i]]')
    # # print(objectClasses[i, 0:objectClassesLength[i]])
    # # print()
    # print()
    # print('input_ids_tmp')
    # print(input_ids_tmp)
    # print()
    # print('lm_label_ids')
    # print(lm_label_ids)
    # print()
    # print()
    # print('input_mask')
    # print(input_mask)
    # print()
    # print('segment_ids')
    # print(segment_ids)






    return input_ids_tmp, lm_label_ids, input_mask, segment_ids





















class XTECDataset:
    def __init__(self, splits: str, distinct=False):
        print()

        self.name = splits
        self.splits = splits.split(',')

        self.objectClasses = json.load(open("data/xtec/object_classes.json"))

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        self.data = []

        for split in self.splits:
            dataTmp = json.load(open("data/xtec/%s.json" % split))


            if distinct:
                # if 'test2015' not in args.test:
                addedImageIds = []
                newDataTmp = []

                print('Distinct data filtering mode:')

                for datumTmp in dataTmp:
                    if datumTmp['img_id'] not in addedImageIds:
                        newDataTmp.append(datumTmp)
                        addedImageIds.append(datumTmp['img_id'])

                print("%d data removed from total of %d in split %s." % (len(dataTmp) - len(newDataTmp), len(dataTmp), split))

                dataTmp = newDataTmp
                del newDataTmp
                # else:
                #     print('There is no image_id duplicates in test sets.')


            if args.maxObjectClassLength > 0 or 1==1:
                newDataTmp = []
                for datum in dataTmp:
                    if str(int(datum['img_id'][-12:])) in self.objectClasses:
                        newDataTmp.append(datum)

                dataTmp = newDataTmp
                del newDataTmp



            if (topk == None):
                self.data.extend(dataTmp)
            else:
                self.data.extend(dataTmp[:topk])

        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # self.id2datum = {
        #     datum['caption_id']: datum
        #     for datum in self.data
        # }






        # self.word2label = json.load(open("data/xtec/trainval_word2label.json"))
        # self.label2word = json.load(open("data/xtec/trainval_label2word.json"))
        # assert len(self.word2label) == len(self.label2word)



    @property
    def num_words(self):
        return len(self.word2label)



    def __len__(self):
        return len(self.data)



    def getObjectClasses(self, imageIds):
        objectClasses = torch.zeros([len(imageIds), args.maxObjectClassLength], dtype=torch.long)

        for i, imageId in enumerate(imageIds):
            imageId = str(int(imageId[-12:]))

            if imageId in self.objectClasses:
                objectClass = (tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.objectClasses[imageId])))[:args.maxObjectClassLength-1] + [sepTokenId]
                objectClass = np.pad(np.array(objectClass), (0,args.maxObjectClassLength - len(objectClass)), 'constant')
                objectClasses[i] = torch.tensor(objectClass, dtype=torch.long)

        return objectClasses






class XTECTorchDataset(Dataset):
    def __init__(self, dataset: XTECDataset):
        super().__init__()

        self.maxCaptionLength = MAX_CAPTION_LENGTH

        self.raw_dataset = dataset

        # if args.tiny:
        #     topk = TINY_IMG_NUM
        # elif args.fast:
        #     topk = FAST_IMG_NUM
        # else:
        #     topk = None
        #
        # img_data = []
        # for split in dataset.splits:
        #     load_topk = 5000 if (split == 'minival' and topk is None) else topk
        #     img_data.extend(load_obj_tsv(
        #         os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
        #         topk=load_topk))
        #
        # self.imgid2img = {}
        # for img_datum in img_data:
        #     self.imgid2img[img_datum['img_id']] = img_datum
        #
        # self.data = []
        # for datum in self.raw_dataset.data:
        #     if datum['img_id'] in self.imgid2img:
        #         self.data.append(datum)

        self.data = self.raw_dataset.data

        print("Use %d data in torch dataset" % (len(self.data)))
        print()



    def __len__(self):
        return len(self.data)



    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        caption = datum['caption']


        # cl = datum['cl']

        # img_info = self.imgid2img[img_id]
        # obj_num = img_info['num_boxes']
        # feats = img_info['features'].copy()
        # boxes = img_info['boxes'].copy()
        # assert obj_num == len(boxes) == len(feats)
        #
        # img_h, img_w = img_info['img_h'], img_info['img_w']
        # boxes = boxes.copy()
        # boxes[:, (0, 2)] /= img_w
        # boxes[:, (1, 3)] /= img_h
        # np.testing.assert_array_less(boxes, 1+1e-5)
        # np.testing.assert_array_less(-boxes, 0+1e-5)

        # target = torch.zeros(self.maxCaptionLength, dtype=torch.long)
        #
        # captionWords = caption.split(' ')
        # # captionLength = len(captionWords)
        #
        # for wordIndex in range(captionWords.index('<E>')):
        #     target[wordIndex] = self.raw_dataset.word2label[captionWords[wordIndex + 1]]
        #
        # # print(captionWords)
        # # print(target)
        # #
        # # exit()






        # train_features = [convert_example_to_features(example, self.args.maxCaptionLength, tokenizer)
        #                   for example in examples]







        # feats, boxes = load_image_features(img_id)[0]
        # feats, boxes = 0, 1


        return caption, img_id


        # return img_id, caption, cl, target






class XTECEvaluator:
    def __init__(self, dataset: XTECDataset):
        self.dataset = dataset



    # def evaluate(self, logits, targets):
    #     score = 0.
    #
    #     for idx, logit in enumerate(logits):
    #         for jdx, word in enumerate(logit):
    #
    #
    #             predictedWordIndex = logit.argmax()
    #
    #             if predictedWordIndex == targets[idx]:
    #                 score += 1
    #
    #     return score / len(targets)



    def dump_result(self, predictions, targets, path, epoch=None):
        suffix = ''

        if epoch is not None:
            suffix = '-epoch' + str(epoch)

        if '.json' in path:
            fileName = path
        else:
            fileName = path + suffix + '.json'

        with open(fileName, 'w') as f:
            result = []

            data = []
            for i in range(len(predictions)):
                for wordIndex in range(len(predictions[i])):
                    predictedWord = tokenizer.convert_ids_to_tokens([predictions[i][wordIndex].item()])
                    targetWord = tokenizer.convert_ids_to_tokens([targets[i][wordIndex].item()])


                    # if targetWord[0] == '.':
                    if targetWord[0] == '[PAD]' and predictedWord[0] == '[PAD]':
                        break

                    data.append(predictedWord[0] + " => " + targetWord[0])

                result.append(data)
                data = []

            json.dump(result, f, indent=4, sort_keys=True, ensure_ascii=False)



    def dump(self, data, path, epoch=None):
        suffix = ''

        if epoch is not None:
            suffix = '-epoch' + str(epoch)

        if '.json' in path:
            fileName = path
        else:
            fileName = path + suffix + '.json'

        with open(fileName, 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)


