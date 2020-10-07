#!/usr/bin/python3

from conllu import parse
import pyconll

conll = pyconll.load_from_file('converted_data.conll')
for sentence in conll:
	for word in sentence:
		print(word.misc)