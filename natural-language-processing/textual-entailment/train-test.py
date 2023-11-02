from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer



# hyperparameters
epochs = 30
batch_size = 16
modelName = 'xlm-roberta-base'
modelSaveName = 'XLM_ROBERTA'
# modelName = 'persiannlp/parsbert-base-parsinlu-entailment'
# modelSaveName = 'PARSBERT'




seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)




device = 'cuda' if torch.cuda.is_available() else 'cpu'





# metric calculation
def calculateF1(preds, labels):
    predictionsFlat = np.argmax(preds, axis=1).flatten()
    labelsFlat = labels.flatten()
    return f1_score(labelsFlat, predictionsFlat, average='weighted')


def calculateAccuracy(preds, labels):
    labelsInverse = {v: k for k, v in labelDictionary.items()}

    predictionsFlat = np.argmax(preds, axis=1).flatten()
    labelsFlat = labels.flatten()

    accuracies = {}
    for label in np.unique(labelsFlat):
        yPredictions = predictionsFlat[labelsFlat==label]
        yActual = labelsFlat[labelsFlat==label]
        print(f'Class: {labelsInverse[label]}: {len(yPredictions[yPredictions==label])/len(yActual)}')
        accuracies[labelsInverse[label]] = len(yPredictions[yPredictions==label])/len(yActual)

    return accuracies



def evaluate(dataloader):

    model.eval()

    totalValidationLoss = 0
    predictions, yActual = [], []

    for batch in dataloader:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        totalValidationLoss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        yActual.append(label_ids)

    validationLossAverage = totalValidationLoss/len(dataloader)

    predictions = np.concatenate(predictions, axis=0)
    yActual = np.concatenate(yActual, axis=0)

    return validationLossAverage, predictions, yActual





# read and process dataset
trainDF = pd.read_csv('q1/train.csv')
validationDF = pd.read_csv('q1/dev.csv')
testDF = pd.read_csv('q1/test.csv')


# trainDF['sent1'] = trainDF['sent1'].str.replace('[^\w\s]','')
# trainDF['sent1'] = trainDF['sent1'].replace('\n',' ', regex=True)
# trainDF['sent2'] = trainDF['sent2'].str.replace('[^\w\s]','')
# trainDF['sent2'] = trainDF['sent2'].replace('\n',' ', regex=True)
#
# validationDF['sent1'] = validationDF['sent1'].str.replace('[^\w\s]','')
# validationDF['sent1'] = validationDF['sent1'].replace('\n',' ', regex=True)
# validationDF['sent2'] = validationDF['sent2'].str.replace('[^\w\s]','')
# validationDF['sent2'] = validationDF['sent2'].replace('\n',' ', regex=True)
#
# testDF['sent1'] = testDF['sent1'].str.replace('[^\w\s]','')
# testDF['sent1'] = testDF['sent1'].replace('\n',' ', regex=True)
# testDF['sent2'] = testDF['sent2'].str.replace('[^\w\s]','')
# testDF['sent2'] = testDF['sent2'].replace('\n',' ', regex=True)


trainDF['sentence'] = trainDF['sent1'] + '[SEP]' + trainDF['sent2']
validationDF['sentence'] = validationDF['sent1'] + '[SEP]' + validationDF['sent2']
testDF['sentence'] = testDF['sent1'] + '[SEP]' + testDF['sent2']

allLabels = trainDF.label.unique()

labelDictionary = {}

for index, label in enumerate(allLabels):
    labelDictionary[label] = index

trainDF['label'] = trainDF.label.replace(labelDictionary)
validationDF['label'] = validationDF.label.replace(labelDictionary)
testDF['label'] = testDF.label.replace(labelDictionary)







# tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelName, do_lower_case=True)

trainDataEncoded = tokenizer.batch_encode_plus(
    trainDF.sentence.values.tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=256,
    truncation=True,
    return_tensors='pt'
)

validationDataEncoded = tokenizer.batch_encode_plus(
    validationDF.sentence.values.tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=256,
    truncation=True,
    return_tensors='pt'
)

testDataEncoded = tokenizer.batch_encode_plus(
    testDF.sentence.values.tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=256,
    truncation=True,
    return_tensors='pt'
)





# prepare datasets and dataloaders
input_ids_train = trainDataEncoded['input_ids']
attention_masks_train = trainDataEncoded['attention_mask']
labels_train = torch.tensor(trainDF.label.values)

input_ids_val = validationDataEncoded['input_ids']
attention_masks_val = validationDataEncoded['attention_mask']
labels_val = torch.tensor(validationDF.label.values)

input_ids_test = testDataEncoded['input_ids']
attention_masks_test = testDataEncoded['attention_mask']
labels_test = torch.tensor(testDF.label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

trainDataloader = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
validationDataloader = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)
testDataloader = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=batch_size)







# prepare model
model = AutoModelForSequenceClassification.from_pretrained(modelName, num_labels=len(labelDictionary), output_attentions=False, output_hidden_states=False)

model.to(device)

print(model)

optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(trainDataloader)*epochs)





# train the model
bestValLoss = 99999
bestEpoch = 0

trainLosses = []
validationLosses = []
validationAccuraciesPerClass = []
validationF1s = []

for epoch in tqdm(range(1, epochs+1)):

    model.train()

    loss_train_total = 0

    progress = tqdm(trainDataloader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress:

        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {
            'input_ids':      batch[0],
            'attention_mask': batch[1],
            'labels':         batch[2],
        }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'q1/finetuned_'+ modelSaveName +'_epoch_{epoch}.model')

    tqdm.write(f'\nEpoch {epoch}')

    trainLossAverage = loss_train_total/len(trainDataloader)
    trainLosses.append(trainLossAverage)
    tqdm.write(f'Training loss: {trainLossAverage}')

    validationLoss, predictions, yActual = evaluate(validationDataloader)
    validationAccuraciesPerClass.append(calculateAccuracy(predictions, yActual))
    validationLosses.append(validationLoss)

    validationF1 = calculateF1(predictions, yActual)
    validationF1s.append(validationF1)

    tqdm.write(f'Validation loss: {validationLoss}')
    tqdm.write(f'Validation F1 Score (Weighted): {validationF1}')

    if validationLoss < bestValLoss:
        torch.save(model.state_dict(), f'q1/finetuned_'+ modelSaveName +'_BEST.model')
        bestValLoss = validationLoss
        bestEpoch = epoch


print()

print('Best validation loss in epoch: ' + str(bestEpoch))






# plot results
accuraciesSorted = {
    'e': [],
    'c': [],
    'n': [],
}

for epochAccuracy in validationAccuraciesPerClass:
    for classs in epochAccuracy:
        accuraciesSorted[classs].append(epochAccuracy[classs])


plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.plot(trainLosses, color='red', marker='.', label='Train Loss')
plt.plot(validationLosses, color='blue', marker='.', label='Validation Loss')
plt.legend()
plt.show()

plt.xlabel('Epoch')
plt.ylabel('Validation F1 Plot')
plt.title('F1 Score')
plt.plot(validationF1s, color='blue', marker='.', label='Validation F1')
plt.legend()
plt.show()

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Per Class Plot')
plt.plot(accuraciesSorted['e'], color='green', marker='.', label='Class e')
plt.plot(accuraciesSorted['c'], color='red', marker='.', label='Class c')
plt.plot(accuraciesSorted['n'], color='blue', marker='.', label='Class n')
plt.legend()
plt.show()







# Test Model using the best saved model in validation
print()

model = AutoModelForSequenceClassification.from_pretrained(modelName, num_labels=len(labelDictionary), output_attentions=False, output_hidden_states=False)

model.to(device)

model.load_state_dict(torch.load('q1/finetuned_'+ modelSaveName +'_BEST.model', map_location=torch.device('cpu')))

_, predictions, yActual = evaluate(testDataloader)
testF1 = calculateF1(predictions, yActual)
calculateAccuracy(predictions, yActual)

tqdm.write(f'Test F1 Score (Weighted): {testF1}')
