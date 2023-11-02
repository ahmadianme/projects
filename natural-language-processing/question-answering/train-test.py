import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time
import math
from tqdm import tqdm
from transformers import AutoTokenizer,AdamW,BertForQuestionAnswering
import matplotlib.pyplot as plt
import string, re




# hyperparameters
# modelName = 'pquad'
# modelName = 'persianqa'
modelName = 'parsquad'
# modelName = 'pquad-persianqa'

transformerName = 'HooshvareLab/bert-base-parsbert-uncased'
epochs = 4
learningRate = 5e-5
batchSize = 12









# download the datasets with the commands below
# wget https://ahmadian.me/qa/pquad/train_samples.json -O pquad/train_samples.json
# wget https://ahmadian.me/qa/pquad/validation_samples.json -O pquad/validation_samples.json
# wget https://ahmadian.me/qa/pquad/test_samples.json -O pquad/test_samples.json
#
# mkdir persianqa
# wget https://ahmadian.me/qa/persianqa/pqa_train.json -O persianqa/pqa_train.json
# wget https://ahmadian.me/qa/persianqa/pqa_test.json -O persianqa/pqa_test.json
#
# mkdir parsquad
# wget https://ahmadian.me/qa/parsquad/ParSQuAD-manual-train.json -O parsquad/ParSQuAD-manual-train.json
# wget https://ahmadian.me/qa/parsquad/ParSQuAD-manual-dev.json -O parsquad/ParSQuAD-manual-dev.json








device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')





paths = []

if modelName == 'pquad':
    paths.append(Path('pquad/train_samples.json'))
elif modelName == 'persianqa':
    paths.append(Path('persianqa/pqa_train.json'))
elif modelName == 'parsquad':
    paths.append(Path('parsquad/ParSQuAD-manual-train.json'))
elif modelName == 'pquad-persianqa':
    paths.append(Path('pquad/train_samples.json'))
    paths.append(Path('persianqa/pqa_train.json'))

if len(paths) > 0:
    texts = []
    queries = []
    answers = []

    for path in paths:
        with open(path, 'rb') as f:
            squad_dict = json.load(f)

        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        texts.append(context)
                        queries.append(question)
                        answers.append(answer)

train_texts, train_queries, train_answers = texts, queries, answers

print(len(texts))
print(len(queries))
print(len(answers))




paths = []

if modelName == 'pquad':
    paths.append(Path('pquad/validation_samples.json'))
elif modelName == 'persianqa':
    selectCount = math.floor(len(train_texts) * 0.15)
    removeCount = len(train_texts) - selectCount
    texts, queries, answers = train_texts[-selectCount:], train_queries[-selectCount:], train_answers[-selectCount:]
    train_texts, train_queries, train_answers = train_texts[:removeCount], train_queries[:removeCount], train_answers[:removeCount]
elif modelName == 'parsquad':
    paths.append(Path('parsquad/ParSQuAD-manual-dev.json'))
elif modelName == 'pquad-persianqa':
    paths.append(Path('pquad/validation_samples.json'))
    # paths.append(Path('persianqa/pqa_train.json'))

if len(paths) > 0:
    texts = []
    queries = []
    answers = []

    for path in paths:
        with open(path, 'rb') as f:
            squad_dict = json.load(f)

        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        texts.append(context)
                        queries.append(question)
                        answers.append(answer)

if modelName == 'pquad-persianqa':
    selectCount = math.floor(len(train_texts) * 0.08)
    removeCount = len(train_texts) - selectCount
    texts += train_texts[-selectCount:]
    queries += train_queries[-selectCount:]
    answers += train_answers[-selectCount:]
    train_texts, train_queries, train_answers = train_texts[:removeCount], train_queries[:removeCount], train_answers[:removeCount]


val_texts, val_queries, val_answers = texts, queries, answers

print(len(texts))
print(len(queries))
print(len(answers))




paths = []

if modelName == 'pquad':
    paths.append(Path('pquad/test_samples.json'))
elif modelName == 'persianqa':
    paths.append(Path('persianqa/pqa_test.json'))
elif modelName == 'parsquad':
    selectCount = math.floor(len(val_texts) * 0.4)
    removeCount = len(val_texts) - selectCount
    texts, queries, answers = val_texts[-selectCount:], val_queries[-selectCount:], val_answers[-selectCount:]
    val_texts, val_queries, val_answers = val_texts[:removeCount], val_queries[:removeCount], val_answers[:removeCount]
elif modelName == 'pquad-persianqa':
    paths.append(Path('pquad/test_samples.json'))
    paths.append(Path('persianqa/pqa_test.json'))

if len(paths) > 0:
    texts = []
    queries = []
    answers = []

    for path in paths:
        with open(path, 'rb') as f:
            squad_dict = json.load(f)

        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        texts.append(context)
                        queries.append(question)
                        answers.append(answer)

test_texts, test_queries, test_answers = texts, queries, answers




print(len(texts))
print(len(queries))
print(len(answers))

print(len(train_texts))
print(len(train_queries))
print(len(train_answers))
print()
print(len(val_texts))
print(len(val_queries))
print(len(val_answers))
print()
print(len(test_texts))
print(len(test_queries))
print(len(test_answers))

print()
print(str(len(train_answers) + len(val_answers) + len(test_answers)))









for answer, text in zip(train_answers, train_texts):
    real_answer = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(real_answer)
    answer['answer_end'] = end_idx

    if text[start_idx:end_idx] == real_answer:
        answer['answer_end'] = end_idx
    elif text[start_idx-1:end_idx-1] == real_answer:
        answer['answer_start'] = start_idx - 1
        answer['answer_end'] = end_idx - 1
    elif text[start_idx-2:end_idx-2] == real_answer:
        answer['answer_start'] = start_idx - 2
        answer['answer_end'] = end_idx - 2



for answer, text in zip(val_answers, val_texts):
    real_answer = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(real_answer)
    answer['answer_end'] = end_idx

    if text[start_idx:end_idx] == real_answer:
        answer['answer_end'] = end_idx
    elif text[start_idx-1:end_idx-1] == real_answer:
        answer['answer_start'] = start_idx - 1
        answer['answer_end'] = end_idx - 1
    elif text[start_idx-2:end_idx-2] == real_answer:
        answer['answer_start'] = start_idx - 2
        answer['answer_end'] = end_idx - 2



for answer, text in zip(test_answers, test_texts):
    real_answer = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(real_answer)
    answer['answer_end'] = end_idx

    if text[start_idx:end_idx] == real_answer:
        answer['answer_end'] = end_idx
    elif text[start_idx-1:end_idx-1] == real_answer:
        answer['answer_start'] = start_idx - 1
        answer['answer_end'] = end_idx - 1
    elif text[start_idx-2:end_idx-2] == real_answer:
        answer['answer_start'] = start_idx - 2
        answer['answer_end'] = end_idx - 2










tokenizer = AutoTokenizer.from_pretrained(transformerName)

train_encodings = tokenizer(train_texts, train_queries, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, val_queries, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, test_queries, truncation=True, padding=True)








def calculateStartEnd(encodings, answers):
  start_positions = []
  end_positions = []

  count = 0

  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

    if start_positions[-1] is None:
      start_positions[-1] = tokenizer.model_max_length

    if end_positions[-1] is None:
      end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1)

      if end_positions[-1] is None:
        count += 1
        end_positions[-1] = tokenizer.model_max_length

  print(count)

  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

calculateStartEnd(train_encodings, train_answers)
calculateStartEnd(val_encodings, val_answers)
calculateStartEnd(test_encodings, test_answers)








class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)







train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)
test_dataset = SquadDataset(test_encodings)







train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True)






model = BertForQuestionAnswering.from_pretrained(transformerName).to(device)
optim = AdamW(model.parameters(), lr=learningRate)


train_losses = []
val_losses = []

print('Training and validating...')

for epoch in range(epochs):
  epoch_time = time.time()

  model.train()

  loss_of_epoch = 0

  print("Training...")

  with tqdm(total=len(train_loader)) as pbar:
    for batch_idx,batch in enumerate(train_loader):

        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()
        loss_of_epoch += loss.item()

        if epoch == 0 and batch_idx == 0:
            train_losses.append(loss_of_epoch)

        if (batch_idx+1) % 50 == 0:
            print("Batch {:} / {:}".format(batch_idx+1,len(train_loader))," Loss:", round(loss.item(),1),"")

        pbar.update(1)

  loss_of_epoch /= len(train_loader)
  train_losses.append(loss_of_epoch)


  torch.save(model, modelName + '-e' + str(epoch+1) + '.pth')

  model.eval()
  print()
  print("Validation...")

  loss_of_epoch = 0

  for batch_idx,batch in enumerate(val_loader):

    with torch.no_grad():

      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      start_positions = batch['start_positions'].to(device)
      end_positions = batch['end_positions'].to(device)

      outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
      loss = outputs[0]
      loss_of_epoch += loss.item()

      if epoch == 0 and batch_idx == 0:
            val_losses.append(loss_of_epoch)

    if (batch_idx+1) % 50 == 0:
       print("Batch {:} / {:}".format(batch_idx+1,len(val_loader))," Loss:", round(loss.item(),1),"")

  loss_of_epoch /= len(val_loader)
  val_losses.append(loss_of_epoch)















_,plot = plt.subplots(1,1,figsize=(15,10))

plot.set_title("Train and Validation Losses",size=20)
plot.set_ylabel('Loss', fontsize = 20)
plot.set_xlabel('Epochs', fontsize = 25)
_=plot.plot(train_losses)
_=plot.plot(val_losses)
_=plot.legend(('Train','Validation'),loc='upper right')
plot.show()









def predict(context,query):

  inputs = tokenizer.encode_plus(query, context, return_tensors='pt').to(device)

  outputs = model(**inputs)
  answer_start = torch.argmax(outputs[0])  # get the most likely beginning of answer with the argmax of the score
  answer_end = torch.argmax(outputs[1]) + 1

  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

  return answer










def normalize_text(s):

  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))








def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
  pred_tokens = normalize_text(prediction).split()
  truth_tokens = normalize_text(truth).split()

  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)

  common_tokens = set(pred_tokens) & set(truth_tokens)

  if len(common_tokens) == 0:
    return 0

  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)

  return 2 * (prec * rec) / (prec + rec)











def doPredict(context,query,answer):

  prediction = predict(context,query)
  em_score = compute_exact_match(prediction, answer)
  f1_score = compute_f1(prediction, answer)

  return prediction, f1_score, em_score











print('Testing...')

f1Epochs = [0]
emEpochs = [0]

for epoch in range(epochs):
    print('Testing epoch ' + str(epoch+1) + ' model started...')
    model = torch.load(modelName + '-e' + str(epoch+1) + '.pth', map_location=device)

    f1Total = 0
    emTotal = 0

    with tqdm(total=len(test_texts)) as pbar:
        for index in range(len(test_texts)):

            prediction, f1_score, em_score = doPredict(test_texts[index], test_queries[index], test_answers[index]['text'])
            f1Total += f1_score
            emTotal += em_score

            pbar.update(1)

        f1Total /= len(test_texts)
        emTotal /= len(test_texts)

        f1Epochs.append(f1Total)
        emEpochs.append(emTotal)

        print('Epoch: ' + str(epoch+1) + ' f1: ' + str(f1Total) + ' em: ' + str(emTotal))
        print()









_,plot = plt.subplots(1,1,figsize=(15,10))

plot.set_title("F1 score",size=20)
plot.set_ylabel('F1 Score', fontsize = 20)
plot.set_xlabel('Epochs', fontsize = 25)
_=plot.plot(f1Epochs)
plot.show()




_,plot = plt.subplots(1,1,figsize=(15,10))

plot.set_title("EM score",size=20)
plot.set_ylabel('EM Score', fontsize = 20)
plot.set_xlabel('Epochs', fontsize = 25)
_=plot.plot(emEpochs)
plot.show()
