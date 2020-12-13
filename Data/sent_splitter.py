#!/usr/bin/python3

# Program to split the sentences an lablels from a data set and puts the sentences in a new file.

import csv
from nltk.tokenize import word_tokenize
import nltk

def main():
	outfile = open('finetuning_trial.txt', 'w')
	with open('tsd_trial.csv') as data:
		file =csv.reader(data)
		next(file, None)
		for row in file:
			outfile.write(row[1])


if __name__ == '__main__':
	main()