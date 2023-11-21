import os
import sys
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from sys import getsizeof
import numpy as np

from arguments import args
from qa_answer_table import load_lxmert_qa
from xtec_model import XTECModel
from xtec_data import XTECDataset, XTECTorchDataset, XTECEvaluator, load_image_features, prepareLanguageFeatures, prepareLanguageFeaturesStep, dotTokenId
from torch.nn.utils.rnn import pack_padded_sequence
from arch.tokenization import BertTokenizer
from xtec_evaluate import bulkBleuScore, bulkMeteorScore

from xtec_data import tokenizer

from tools.coco_caption.pycocotools.coco import COCO
from tools.coco_caption.pycocoevalcap.eval import COCOEvalCap

useGPU = torch.cuda.is_available()
device = torch.device('cuda') if useGPU else torch.device('cpu')

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')






def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False, distinct=False) -> DataTuple:
    dset = XTECDataset(splits, distinct)
    tset = XTECTorchDataset(dset)
    evaluator = XTECEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=int(args.num_workers),
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)






class XTEC:
    def __init__(self):
        if args.train != "":
            self.train_tuple = get_data_tuple(
                args.train, bs=args.batch_size, shuffle=True, drop_last=True
            )

        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=64,
                shuffle=False, drop_last=False,
                distinct=True
            )
        else:
            self.valid_tuple = None

        # Model
        self.model = XTECModel()

        if args.load_lxmert is not None:
            self.model.load(args.load_lxmert)

        if useGPU:
            self.model = self.model.cuda()

        if args.multiGPU:
            self.multi_gpu()

        if args.train != "":
            if 'bert' in args.optim:
                batch_per_epoch = len(self.train_tuple.loader)
                t_total = int(batch_per_epoch * args.epochs)
                print("BertAdam Total Iters: %d" % t_total)

                from arch.optimization import BertAdam

                self.optim = BertAdam(list(self.model.parameters()),
                                      lr=args.lr,
                                      warmup=0.1,
                                      t_total=t_total*20)
            else:
                self.optim = args.optimizer(self.model.parameters(), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)







    def trainBatch(self, caption, feats, boxes, objectClasses=None):
        self.optim.zero_grad()

        input_ids = prepareLanguageFeatures(caption)

        lengths = torch.count_nonzero(input_ids, dim=1).to(device) - 1
        maxLength = lengths.max().cpu().data.numpy()

        if maxLength > args.maxCaptionLength - 2:
            maxLength = args.maxCaptionLength - 2

        totalLoss = 0

        predictions = torch.zeros([len(caption), input_ids.size(1)], dtype=torch.int)
        targets = torch.zeros([len(caption), input_ids.size(1)], dtype=torch.int)

        predictionCount = 0
        correctPredictionCount = 0

        for wordPosition in range(maxLength + 1):
            input_ids_tmp, lm_label_ids, input_mask, segment_ids = prepareLanguageFeaturesStep(input_ids, wordPosition, lengths, objectClasses)

            lang_prediction_scores, cross_relationship_score, loss = self.model(input_ids_tmp, segment_ids, input_mask, lm_label_ids, feats, boxes, mode='train')




            for i in range(len(caption)):
                prediction = lang_prediction_scores[i][wordPosition + 1].argmax().item()
                target = lm_label_ids[i][wordPosition + 1].item()

                if target != -1 and target != 0:
                    predictionCount += 1

                    if prediction == target:
                        correctPredictionCount += 1




            if args.multiGPU:
                loss = loss.mean()

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optim.step()

            totalLoss += loss.item()

        accuracy = correctPredictionCount / predictionCount

        return totalLoss, totalLoss / maxLength, accuracy






    def predictBatch(self, feats, boxes, objectClasses=None):
        input_ids = torch.tensor([[0] * args.maxCaptionLength] * len(feats), dtype=torch.long).to(device)

        lengths = torch.full([len(feats)], 1, dtype=torch.int).to(device) - 1
        maxLength = args.maxCaptionLength - 2

        predictions = torch.zeros([len(feats), args.maxCaptionLength], dtype=torch.int)

        for wordPosition in range(maxLength):
            input_ids_tmp, _, input_mask, segment_ids = prepareLanguageFeaturesStep(input_ids, wordPosition, lengths, objectClasses)

            lang_prediction_scores, _ = self.model(input_ids_tmp, segment_ids, input_mask, None, feats, boxes, mode='predict')

            allPredictionsDone = True
            for i in range(len(feats)):
                if lengths[i] < wordPosition:
                    continue;

                prediction = lang_prediction_scores[i][wordPosition + 1].argmax().item()

                if prediction != 0:
                    predictions[i][wordPosition] = prediction
                    input_ids[i][wordPosition] = prediction


                if prediction != dotTokenId:
                    lengths[i] += 1
                    allPredictionsDone = False

            if allPredictionsDone:
                break;

        return predictions






    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple

        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            epochTotalLoss = 0;
            epochTotalAccuracy = 0;

            for i, (caption, img_id) in iter_wrapper(enumerate(loader)):
                self.model.train()

                feats, boxes = load_image_features(img_id)
                feats, boxes = feats.to(device), boxes.to(device)

                objectClasses = None
                if args.maxObjectClassLength > 0:
                    objectClasses = dset.getObjectClasses(img_id)

                loss, averateLoss, accuracy = self.trainBatch(caption, feats, boxes, objectClasses)

                epochTotalLoss += averateLoss
                epochTotalAccuracy += accuracy

            log_str = "\nEpoch %d: %0.3f => Train Total Loss\n" % (epoch + 1, averateLoss) + \
                      "Epoch %d: %0.3f => Train Average Loss\n" % (epoch + 1, epochTotalLoss / len(self.train_tuple.loader)) + \
                      "Epoch %d: %0.3f => Train Accuracy (Percent)\n" % (epoch + 1, epochTotalAccuracy / len(self.train_tuple.loader) * 100.)

            scores = {}
            if self.valid_tuple is not None:
                scores = self.evaluate(eval_tuple, args.output + '/train_eval', epoch=epoch+1, accuracy=True)
                if scores['accuracy'] > best_valid:
                    best_valid = scores['accuracy']
                    self.save("BEST")

                log_str += "Epoch %d: %0.3f => Validation Accuracy (Percent)\n" % (epoch + 1, scores['accuracy'] * 100.) + \
                           "Epoch %d: %0.3f => Best Validation Accuracy (Percent)\n" % (epoch + 1, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/train.log", 'a') as f:
                f.write(log_str + "\n" + str(scores) + "\n")
                f.flush()

            self.save("epoch" + str(epoch+1))








    def predict(self, eval_tuple: DataTuple, dump=None, epoch=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple

        allPredictions = None
        allTargets = None
        allImageIds = None


        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        for i, (captions, imageIds) in iter_wrapper(enumerate(loader)):

            feats, boxes = load_image_features(imageIds)
            feats, boxes = feats.to(device), boxes.to(device)

            objectClasses = None
            if args.maxObjectClassLength > 0:
                objectClasses = dset.getObjectClasses(imageIds)

            targets = prepareLanguageFeatures(captions)

            with torch.no_grad():
                feats, boxes = feats.to(device), boxes.to(device)

                predictions = self.predictBatch(feats, boxes, objectClasses)

                if allPredictions == None:
                    allPredictions = predictions
                    allTargets = targets
                    allImageIds = imageIds
                else:
                    allPredictions = torch.cat((allPredictions, predictions), 0)
                    allTargets = torch.cat((allTargets, targets), 0)
                    allImageIds += imageIds

        allPredictions = allPredictions.cpu().numpy()
        allTargets = allTargets.cpu().numpy()

        if dump is not None:
            evaluator.dump_result(allPredictions, allTargets, dump + '_prediction_pairs', epoch)

        return allPredictions, allTargets, allImageIds





    def evaluate(self, eval_tuple: DataTuple, dump=None, epoch=None, accuracy=False):
        predictions, targets, imgIds = self.predict(eval_tuple, dump=dump, epoch=epoch)

        scores = {}

        if accuracy:
            totalPredictions = 0
            totalCorrectPredictions = 0

            for i in range(len(predictions)):
                for wordPosition in range(len(predictions[i])):
                    if predictions[i][wordPosition] and targets[i][wordPosition] == 0:
                        break

                    totalPredictions += 1

                    if targets[i][wordPosition] == predictions[i][wordPosition]:
                        totalCorrectPredictions += 1

            scores['accuracy'] = totalCorrectPredictions / totalPredictions


        targetsStrings = []
        predictionsStrings = []

        print(len(imgIds))
        print(len(predictions))
        print(len(targets))

        exceptionMessages = []
        for i in range(len(targets)):
            try:
                targetsStrings.append({'image_id': int(imgIds[i][-12:]), 'caption': (' '.join(tokenizer.convert_ids_to_tokens(np.trim_zeros(targets[i])[:-1])) + '.').replace(' ##', '').replace('##', '')})
                predictionsStrings.append({'image_id': int(imgIds[i][-12:]), 'caption': (' '.join(tokenizer.convert_ids_to_tokens(np.trim_zeros(predictions[i])[:-1])) + '.').replace(' ##', '').replace('##', '')})
            except Exception as e:
                exceptionMessages.append({'target': (' '.join(tokenizer.convert_ids_to_tokens(np.trim_zeros(targets[i])[:-1])) + '.').replace(' ##', '').replace('##', ''), 'prediction': (' '.join(tokenizer.convert_ids_to_tokens(np.trim_zeros(predictions[i])[:-1])) + '.').replace(' ##', '').replace('##', '')})
                print('Exception occured while converting prediction strings for score calculation.')

        suffix = ''
        if epoch is not None:
            suffix = '-epoch'+str(epoch)

        targetsFile = dump + '_targets'+suffix+'.json'
        predictionsFile = dump + '_predictions'+suffix+'.json'

        eval_tuple.evaluator.dump(targetsStrings, targetsFile)
        eval_tuple.evaluator.dump(predictionsStrings, predictionsFile)

        scores = dict(scores, **self.calculateScores(targetsFile, predictionsFile))
        eval_tuple.evaluator.dump(scores, dump + '_scores'+suffix+'.json')

        eval_tuple.evaluator.dump(exceptionMessages, dump + '_exceptions'+suffix+'.json')

        return scores






    def calculateScores(self, targets, predictions):
        scores = {}

        if args.test == 'test2015':
            annotation_filepath = './src/tools/coco_caption/annotations/image_info_test2015.json'
        elif args.test == 'test2015dev':
            annotation_filepath = './src/tools/coco_caption/annotations/image_info_test-dev2015.json'
        else:
            annotation_filepath = './src/tools/coco_caption/annotations/captions_val2014.json'

        print(annotation_filepath)

        # create coco object and cocoRes object
        coco = COCO(annotation_filepath)
        cocoRes = coco.loadRes(os.getcwd() + '/' + predictions)

        # create cocoEval object by taking coco and cocoRes
        cocoEval = COCOEvalCap(coco, cocoRes)

        # please remove this line when evaluating the full validation set
        cocoEval.params['image_id'] = cocoRes.getImgIds()

        # evaluate results
        # SPICE will take a few minutes the first time, but speeds up due to caching
        try:
            cocoEval.evaluate()
        except Exception as e:
            print('Error in XTEC score calculation!')
            print(str(e))

        for metric, score in cocoEval.eval.items():
            scores[metric] = score

        return scores






    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))





    def load(self, path):
        print("Load model from %s" % path)

        if useGPU:
            state_dict = torch.load("%s.pth" % path)
        else:
            state_dict = torch.load("%s.pth" % path, map_location=torch.device('cpu'))

        self.model.load_state_dict(state_dict)









def multi_gpu(self):
    self.model = nn.DataParallel(self.model)




if __name__ == "__main__":
    if (useGPU):
        print('Starting XTEC (Cross-modal Transformer Encoder Captioner) using GPU (Cuda)')
    else:
        print('Starting XTEC (Cross-modal Transformer Encoder Captioner) using CPU')

    print('')

    xtec = XTEC()

    if args.load is not None:
        xtec.load(args.load)

    if args.test is not None:
        if 'test' in args.test:
            result = xtec.evaluate(
                get_data_tuple(args.test, bs=1000, shuffle=False, drop_last=False, distinct=True),
                dump=os.path.join(args.output, args.test),
                accuracy=False,
            )
        else:
            result = xtec.evaluate(
                get_data_tuple(args.test, bs=1000, shuffle=False, drop_last=False, distinct=True),
                dump=os.path.join(args.output, args.test),
                accuracy=True,
            )
    else:
        print('Splits in Train data:', xtec.train_tuple.dataset.splits)

        if xtec.valid_tuple is not None:
            print('Splits in Valid data:', xtec.valid_tuple.dataset.splits)
        else:
            print("DO NOT USE VALIDATION")

        if xtec.train_tuple is not None:
            xtec.train(xtec.train_tuple, xtec.valid_tuple)


