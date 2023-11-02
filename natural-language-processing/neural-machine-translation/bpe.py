trainEnFile = 'data/train.en'
trainFaFile = 'data/train.fa'
validEnFile = 'data/valid.en'
validFaFile = 'data/valid.fa'
testEnFile = 'data/test.en'
testFaFile = 'data/test.fa'



trainEnFileProc = trainEnFile + '.proc'
trainFaFileProc = trainFaFile + '.proc'
validEnFileProc = validEnFile + '.proc'
validFaFileProc = validFaFile + '.proc'
testEnFileProc = testEnFile + '.proc'
testFaFileProc = testFaFile + '.proc'





import pyonmttok

args = {
    "mode": "aggressive",
    "joiner_annotate": True,
    "preserve_placeholders": True,
    "case_markup": True,
    "soft_case_regions": True,
    "preserve_segmented_tokens": True,
}
n_symbols = 40000



tokenizer_default = pyonmttok.Tokenizer(**args)
learner = pyonmttok.BPELearner(tokenizer=tokenizer_default, symbols=n_symbols)
# load training corpus
learner.ingest_file(trainEnFileProc)

# learn and store bpe model
tokenizer = learner.learn("opennmt/en.bpe")

tokenizer.tokenize_file(f"{trainEnFileProc}", f"{trainEnFile}.bpe")
tokenizer.tokenize_file(f"{validEnFileProc}", f"{validEnFile}.bpe")
tokenizer.tokenize_file(f"{testEnFileProc}", f"{testEnFile}.bpe")



tokenizer_default = pyonmttok.Tokenizer(**args)
learner = pyonmttok.BPELearner(tokenizer=tokenizer_default, symbols=n_symbols)
# load training corpus
learner.ingest_file(trainFaFileProc)

# learn and store bpe model
tokenizer = learner.learn("opennmt/fa.bpe")

tokenizer.tokenize_file(f"{trainFaFileProc}", f"{trainFaFile}.bpe")
tokenizer.tokenize_file(f"{validFaFileProc}", f"{validFaFile}.bpe")
tokenizer.tokenize_file(f"{testFaFileProc}", f"{testFaFile}.bpe")



trainEnFileBPE = trainEnFile + '.bpe'
trainFaFileBPE = trainFaFile + '.bpe'
validEnFileBPE = validEnFile + '.bpe'
validFaFileBPE = validFaFile + '.bpe'
testEnFileBPE = testEnFile + '.bpe'
testFaFileBPE = testFaFile + '.bpe'
