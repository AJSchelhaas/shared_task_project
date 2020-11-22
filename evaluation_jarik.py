#!/urs/bin/python3 
import pandas as pd
import json
import re
from statistics import mean 

def get_spans(gold, pred):

	# Get spans for gold data
	gold_data = pd.read_csv(gold).spans
	SG = []
	for span in gold_data:
		span = json.loads(span)
		SG.append(span)
	
	# Get predicted spans
	with open(pred,'r') as pred_data:
		lines = pred_data.readlines()
		SA = []
		for line in lines:
			try:
				span = re.search('"\[(.*?)\]"', line).group(1)
				span = '['+span+']'
				span = json.loads(span)
				SA.append(span)
			except:
				pass

	print("SG: {}".format(len(SG)))
	print("SA: {}".format(len(SA)))

	return SG, SA

def calculate_F1(SG, SA):

	F1_scores = []

	for i in range(len(SG)):

		StG = SG[i]
		StA = SA[i]

		print(StG)
		print(StA)

		if StG == [] and StA == []:
			print("AND")
			F1_scores.append(1)
		elif StG == [] or StA == [] and StG != StA:
			print("XOR")
			F1_scores.append(0)
		else:
			print("NOR")
			intersection = list(set(StA) & set(StG))

			P = len(intersection)/len(StA)
			R = len(intersection)/len(StG)

			F1 = (2*(P*R))/(P+R)
			F1_scores.append(F1)

		print("\n\n")

	F1_score = mean(F1_scores)

	return F1_score


def main():

	SG, SA = get_spans("tsd_train.csv", "results.csv")

	F1_score = calculate_F1(SG, SA)
	print("System performance (avg. F1): {}".format(F1_score))


if __name__ == '__main__':
	main()
