#!/usr/bin/python3

import csv
import ast

def read_corpus():
	with open('tsd_train.csv', 'r') as data:
		file = csv.reader(data)
		next(file, None)
		for row in file:
			span_string = row[0].strip('][').split(', ')
			text = row[1]
			span = []
			if span_string == ['']:
				span = span
			else:
				for e in span_string:
					span.append(int(e))
				print(span)

			if span == []:
				print('FULL TOXIC TEXT')
			else:
				toxic_span = ""
				for i in span:
					toxic_span += text[i]
				print(toxic_span)


def main():

	read_corpus()

if __name__ == '__main__':
	main()