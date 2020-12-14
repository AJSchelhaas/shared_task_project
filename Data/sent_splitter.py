#!/usr/bin/python3

# Program to split the sentences an lablels from a data set and puts the sentences in a new file.

import csv
import json

def main():

# 	with open('tsd_train.csv') as data:
# 		with open('finetuning_train.json', 'w', encoding='utf-8') as outfile:
# 			json.dump(data, outfile, ensure_ascii=False)

	with open('../finetuning/Test/finetuning_train.csv', 'w', encoding = 'utf-8') as outfile:
		with open('Source/tsd_train.csv', encoding = 'utf-8') as data:
			writer = csv.writer(outfile)
			file = csv.reader(data)
			next(file, None)
			for row in file:
				outfile.write(row[1])


if __name__ == '__main__':
	main()