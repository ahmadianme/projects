import matplotlib.pyplot as plt
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# %matplotlib inline

torch.manual_seed(1)


nltk.download('treebank')
nltk.download('universal_tagset')







def prepareSequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)






class RecurrentModel(nn.Module):

    def __init__(self, embedding_dim, recurrent_dim, ff_dim, vocab_size, tagset_size, dropoutRate):
        super(RecurrentModel, self).__init__()
        self.ff_dim = ff_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # self.recurrent = nn.LSTM(embedding_dim, recurrent_dim)
        self.recurrent = nn.LSTM(embedding_dim, recurrent_dim, bidirectional=False, dropout=dropoutRate)
        # self.recurrent = nn.LSTM(embedding_dim, recurrent_dim, bidirectional=True, dropout=dropoutRate)
        # self.recurrent = nn.LSTM(embedding_dim, recurrent_dim, num_layers=2, bidirectional=False, dropout=dropoutRate)

        # The linear layer that maps from hidden state space to tag space
        self.ff1 = nn.Linear(recurrent_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, tagset_size)
        # self.ff1 = nn.Linear(recurrent_dim, tagset_size)

        self.dropout = nn.Dropout(dropoutRate)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.recurrent(embeds.view(len(sentence), 1, -1))
        # tagSpace = self.ff1(lstm_out.view(len(sentence), -1))
        # lstm_out = self.dropout(lstm_out)
        tagSpace = self.ff1(lstm_out)
        # tagSpace = self.dropout(tagSpace)
        tagSpace = self.ff2(tagSpace.view(len(sentence), -1))
        tagScore = F.log_softmax(tagSpace, dim=1)
        return tagScore






def train(trainData, validationData=None):
    validationAccuracies = []
    bestAccuracy = 0
    losses = []

    for epoch in range(epochs):
        epochLosses = []

        with tqdm(total=len(trainData)) as pbar:
            for sentence, tags in trainData:
                model.zero_grad()

                sentence_in = prepareSequence(sentence, wordToIndex).to(device=device)
                targets = prepareSequence(tags, tagToIndex).to(device=device)

                tagScore = model(sentence_in)

                loss = loss_function(tagScore, targets)

                epochLosses.append(loss.item())

                loss.backward()
                optimizer.step()

                pbar.update(1)

            losses.append(np.mean(epochLosses))

        if validationData is not None:
            validationAccuracy = test(validationData)
            validationAccuracies.append(validationAccuracy)

            if (validationAccuracy > bestAccuracy):
                bestAccuracy = validationAccuracy
                torch.save(model.state_dict(), 'q1-nn-best-model.pth')

            print('Epoch ' + str(epoch+1) + ' done. Validation Accuracy: ' + str(validationAccuracy))
            print()

    return validationAccuracies, losses







def test(testData):
    totalCount = 0
    correctCount = 0

    with torch.no_grad():
        for testItem in testData:
            inputs = prepareSequence(testItem[0], wordToIndex).to(device=device)
            tagScore = model(inputs)

            totalCount += len(inputs)

            for i, score in enumerate(tagScore):
                if indexToTag[torch.argmax(score).item()] == testItem[1][i]:
                    correctCount += 1

    return correctCount / totalCount








NLTKDataset = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))


data = []

for item in NLTKDataset:
    sentence, tags = [], []

    for word, tag in item:
        sentence.append(word)
        tags.append(tag)

    data.append((sentence, tags))



trainData, testData = train_test_split(data,train_size=0.80, test_size=0.20, random_state = 101)
trainData, validationData = train_test_split(trainData,train_size=0.85, test_size=0.15, random_state = 101)




wordToIndex = {}
tagToIndex = {}

for sent, tags in data:
    for i, word in enumerate(sent):
        if tags[i] not in tagToIndex:
            tagToIndex[tags[i]] = len(tagToIndex)

        if word not in wordToIndex:  # word has not been assigned an index yet
            wordToIndex[word] = len(wordToIndex)  # Assign each word with a unique index

# print(wordToIndex)



indexToTag = {value:key for key, value in tagToIndex.items()}





device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: ' + device)
print()


# hyperparameters
embeddingDimension = 1000
recurrentDimension = 512
linearDimension = 128
dropoutRate = 0.25
epochs = 32


model = RecurrentModel(embeddingDimension, recurrentDimension, linearDimension, len(wordToIndex), len(tagToIndex), dropoutRate)
# loss_function = nn.NLLLoss()
loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1)
# optimizer = optim.Adam(model.parameters())


model.to(device=device)





print('Training the model...')
print()
accuracies, losses = train(trainData, validationData)



plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Train Loss Plot')
plt.plot(losses, color='red', marker='.')
plt.show()

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Plot')
plt.plot(accuracies, color='blue', marker='.')
plt.show()

print()
print('Testing the model...')
print()

accuracy = test(testData)

print('Test Accuracy: ' + str(accuracy))
