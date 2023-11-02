from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace


# hyperparameters
dataset = 'wiki'
# dataset = 'gutenberg'



saveFiles = {
    'wiki': 'tokenizer-wiki.json',
    'gutenberg': 'tokenizer-gutenberg.json'
}

dataFiles = {
    'wiki': [f"../data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]],
    'gutenberg': ['../data/gutengerg-pg16457.txt']
}

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=30000)

tokenizer.pre_tokenizer = Whitespace()

files = dataFiles[dataset]
tokenizer.train(files, trainer)

tokenizer.save(saveFiles[dataset])

# tokenizer = Tokenizer.from_file(saveFiles[dataset])

output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)

print(output.ids)
