import tensorflow
import numpy as np
import pandas as pd
from collections import Counter
import random
import IPython
from IPython.display import Image, Audio, display
import music21
from music21 import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adamax
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
np.random.seed(42)


import os
from music21 import instrument, note, converter, chord
import pickle



# hyperparameters
dropoutEnabled = True

dp = 'dropout' if dropoutEnabled else 'no-dropout'

targetArtist = 'chopin'
# targetArtist = 'mozart'

projectDir = targetArtist
dataDir = projectDir + 'classical-music-midi/'
dataFile = projectDir + 'data.pkl'
modelFile = projectDir + dp + '/model.h5'
historyFile = projectDir + dp + '/history.pkl'
generated = projectDir + dp + '/generated-' + targetArtist + '.midi'
seq_lenght = 40
epochs = 200






def extract_notes(file):
    notes = []
    pick = None
    for j in file:
        songs = instrument.partitionByInstrument(j)
        for part in songs.parts:
            pick = part.recurse()
            for element in pick:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))

    return notes





def chords_n_notes(Snippet):
    Melody = []
    offset = 0
    for i in Snippet:
        if ("." in i or i.isdigit()):
            chord_notes = i.split(".") #Seperating the notes in chord
            notes = []
            for j in chord_notes:
                inst_note=int(j)
                note_snip = note.Note(inst_note)
                notes.append(note_snip)
                chord_snip = chord.Chord(notes)
                chord_snip.offset = offset
                Melody.append(chord_snip)
        # pattern is a note
        else:
            note_snip = note.Note(i)
            note_snip.offset = offset
            Melody.append(note_snip)
        # increase offset each iteration so that notes do not stack
        offset += 1
    Melody_midi = stream.Stream(Melody)
    return Melody_midi





def Malody_Generator(Note_Count, model):
    # Note_Count = Count of Notes want to generate
    seed = [np.random.randint(0,len(X_test)-1, seq_lenght)] #sequence_length = 40
    Music = ""
    Notes_Generated=[]
    for i in range(Note_Count):
        seed = np.array(seed).reshape(1,seq_lenght,1)
        prediction = model.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1.0 #diversity
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index/ float(L_symb)  # L_symb is length of total unique characters
        Notes_Generated.append(index)
        Music = [reverse_mapping[char] for char in Notes_Generated] # reverse_mapping is a dictionary that maps each number to its related note
        seed = np.insert(seed[0],len(seed[0]),index_N)
        seed = seed[1:]
    #Now, we have music in form or a list of chords and notes and we want to be a midi file.
    Melody = chords_n_notes(Music)
    Melody_midi = stream.Stream(Melody)
    return Music,Melody_midi






def processRawData(artistName):
    files = []
    dirName = dataDir + artistName + '/'

    for filename in os.listdir(dirName):
        f = os.path.join(dirName, filename)
        midi = converter.parse(f)
        files.append(midi)

    data = extract_notes(files)

    return data




def initializeData():
    data = processRawData(targetArtist)
    saveData(data)



def saveData(data):
    pickle.dump(data, open( dataFile, "wb"))



def loadData():
    return pickle.load(open( dataFile, "rb"))


def show(music):
    img = mpimg.imread(music.write("melody.png"))
    plt.imshow(img)
    plt.show()




def removeRaraChords(data):
    rare_note = []
    count_num = Counter(data)
    for index, (key, value) in enumerate(count_num.items()):
        if value < 100:
            m = key
            rare_note.append(m)

    for element in data:
        if element in rare_note:
            data.remove(element)

    return data




def loadHistory():
    return pickle.load(open( historyFile, "rb"))







def makeModel():
    model = Sequential()
    model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(256))
    model.add(Dropout(0.1))
    model.add(Dense(256))
    model.add(Dropout(0.1))
    model.add(Dense(y.shape[1], activation='softmax'))

    opt = Adamax(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    model.summary()

    return model





def trainModel(model):
    history = model.fit(X_train, y_train, batch_size=256, epochs=epochs)

    model.save(modelFile)

    with open(historyFile, 'wb') as file:
        pickle.dump(history.history, file)

    return model







def generateMelody():
    Music_notes, Melody = Malody_Generator(100, model)

    Melody.write('midi', generated)











# initializeData()




data = loadData()
data = removeRaraChords(data)



symb = sorted(list(set(data)))

L_corpus = len(data)
L_symb = len(symb)

mapping = dict((c, i) for i, c in enumerate(symb))
reverse_mapping = dict((i, c) for i, c in enumerate(symb))

length = 40
features = []
targets = []
for i in range(0, L_corpus - length, 1):
    feature = data[i:i + length]
    target = data[i + length]
    features.append([mapping[j] for j in feature])
    targets.append(mapping[target])


L_datapoints = len(targets)

X = (np.reshape(features, (L_datapoints, length, 1)))/ float(L_symb)
y = tensorflow.keras.utils.to_categorical(targets)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)










model = makeModel()
model = trainModel(model)
# exit()




# model = load_model(modelFile)
# model = trainModel(model)
# history = loadHistory()




history_df = pd.DataFrame(history)

plt.plot(range(len(history_df["loss"])), history_df["loss"], label = "Train Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.legend()
plt.show()







