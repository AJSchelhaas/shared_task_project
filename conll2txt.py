import pyconll

conllfile = './Data/converted_data_2.conll'
conll = pyconll.load_from_file(conllfile)

with open('text.txt', 'w') as f:

    for sentence in conll:
        for token in sentence:
            line = ''
            line += token.form+' '
            for x in token.misc['tox']:
                line += x
            line += '\n'
            f.write(line)
        f.write('\n')