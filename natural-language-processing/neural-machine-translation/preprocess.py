trainEnFile = 'data/train.en'
trainFaFile = 'data/train.fa'
validEnFile = 'data/valid.en'
validFaFile = 'data/valid.fa'
testEnFile = 'data/test.en'
testFaFile = 'data/test.fa'




with open(trainEnFile, 'r') as file:
    data = file.read().lower()

text_file = open(trainEnFile + '.proc', "w")
text_file.write(data)
text_file.close()

with open(validEnFile, 'r') as file:
    data = file.read().lower()

text_file = open(validEnFile + '.proc', "w")
text_file.write(data)
text_file.close()

with open(testEnFile, 'r') as file:
    data = file.read().lower()

text_file = open(testEnFile + '.proc', "w")
text_file.write(data)
text_file.close()



with open(trainFaFile, 'r') as file:
    data = file.read().replace('\u200c', ' ')

text_file = open(trainFaFile + '.proc', "w")
text_file.write(data)
text_file.close()

with open(trainMinFaFile, 'r') as file:
    data = file.read().replace('\u200c', ' ')

text_file = open(trainMinFaFile + '.proc', "w")
text_file.write(data)
text_file.close()

with open(validFaFile, 'r') as file:
    data = file.read().replace('\u200c', ' ')

text_file = open(validFaFile + '.proc', "w")
text_file.write(data)
text_file.close()

with open(testFaFile, 'r') as file:
    data = file.read().replace('\u200c', ' ')

text_file = open(testFaFile + '.proc', "w")
text_file.write(data)
text_file.close()





trainEnFileProc = trainEnFile + '.proc'
trainFaFileProc = trainFaFile + '.proc'
validEnFileProc = validEnFile + '.proc'
validFaFileProc = validFaFile + '.proc'
testEnFileProc = testEnFile + '.proc'
testFaFileProc = testFaFile + '.proc'






