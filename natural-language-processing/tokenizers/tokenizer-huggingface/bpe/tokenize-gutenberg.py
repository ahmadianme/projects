from tokenizers import Tokenizer

# hyperparameters
# dataset = 'wiki'
dataset = 'gutenberg'



saveFiles = {
    'wiki': 'tokenizer-wiki.json',
    'gutenberg': 'tokenizer-gutenberg.json'
}


with open('../data/gutengerg-pg16457.txt') as f:
    text = f.read()

    tokenizer = Tokenizer.from_file(saveFiles[dataset])

    print(text + '\n')
    output = tokenizer.encode(text)
    print(str(output.tokens) + '\n')

    print(str(output.ids) + '\n')

    print('Token count: ' + str(len(output.ids)))
