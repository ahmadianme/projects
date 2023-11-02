import pickle
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dropout, LSTM, Dense, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt







mode = 'train'
# mode = 'predict'

projectDir = ''
dataFile = projectDir + 'LYRICS_DATASET.csv'
modelFile = projectDir + 'model.h5'
historyFile = projectDir + 'poisson/history.pkl'
checkpointPath = projectDir
epochs=10
dataPercent = .5






def to_camel_case(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])



def predict(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text









df = pd.read_csv(dataFile)

df.drop(['Artist Name' ,'Song Name'],axis=1,inplace=True)

df['Number_of_words'] = df['Lyrics'].apply(lambda x:len(str(x).split()))

if dataPercent < 1:
    totalCount = round(len(df) * dataPercent)
    df = df[:totalCount]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Lyrics'].astype(str).str.lower())

total_words = len(tokenizer.word_index)+1
tokenized_sentences = tokenizer.texts_to_sequences(df['Lyrics'].astype(str))
tokenized_sentences[0]



input_sequences = list()
for i in tokenized_sentences:
    for t in range(1, len(i)):
        n_gram_sequence = i[:t+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))











if mode == 'train':
    X, labels = input_sequences[:,:-1],input_sequences[:,-1]
    y = tf.keras.utils.to_categorical(labels, num_classes=total_words)



    model = Sequential()
    model.add(Embedding(total_words, 40, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(250)))
    model.add(Dropout(0.1))
    model.add(Dense(250))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC', 'KLDivergence', 'CategoricalCrossentropy', 'TopKCategoricalAccuracy'])
    model.summary()
    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto')

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpointPath,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True)

    history_logger=tf.keras.callbacks.CSVLogger('history_log.csv', separator=",", append=True)


    history = model.fit(X, y, epochs=epochs, verbose=1, callbacks=[checkpoint_callback, history_logger, earlystop])



    model.save(modelFile)

    with open(historyFile, 'wb') as file:
        pickle.dump(history.history, file)



    plt.plot(history.history['loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.legend()
    plt.show()


else:
    model = load_model(modelFile)
    history = pickle.load(open( historyFile, "rb"))

    for metric in history:
        metricCamelCase = to_camel_case(metric)
        metricCamelCase = metricCamelCase.strip().capitalize()

        plt.plot(history[metric], label=metricCamelCase)
        plt.xlabel('Epoch')
        plt.ylabel(metricCamelCase)
        plt.title(metricCamelCase + ' Plot')
        plt.legend()
        plt.show()


    generatedText = predict("", 30)

    print(generatedText)








