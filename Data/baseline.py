#!/usr/bin/python3

import csv
import pyconll
import conllu

def read_corpus():
	with open('tsd_train.csv', 'r') as data:
		file = csv.reader(data)
		next(file, None)
		texts = []
		spans = []
		for row in file:
			span = []
			span_string = row[0].strip('][').split(', ')
			if span_string == ['']:
				spans.append(span)
			else:
				for e in span_string:
					span.append(int(e))
				spans.append(span)

			texts.append(row[1].strip().replace('\n', ' '))

	return spans, texts

def main():

	spans, texts = read_corpus()
	for i in range(len(spans)):
		print("{}\n{}\n\n".format(spans[i], texts[i]))

if __name__ == '__main__':
	main()