#!/urs/bin/python3 
import pandas as pd
import json
import re
from statistics import mean 

def get_spans(gold, pred):
	"""Extracts the spans for the gold data (SG) and the the spans predicted by the model (SA)"""

	# Get spans for gold data
	gold_data = pd.read_csv(gold).spans
	SG = []
	for span in gold_data:
		span = json.loads(span)
		SG.append(span)
	
	# Get predicted spans
	with open(pred,'r',encoding='utf-8') as pred_data:
		lines = pred_data.readlines()
		SA = []
		for line in lines[1:]:
			try:
				span = re.search('"\[(.*?)\]"', line).group(1)
				span = '['+span+']'
				span = json.loads(span)
				SA.append(span)
			except:
				pass

	return SG, SA

def calculate_F1(SG, SA):
	"""Calculates the F1-score per instance t and from this determines an average F1-score for the performance of the model"""

	F1_scores = []

	for i in range(len(SG)):

		StG = SG[i]
		StA = SA[i]

		print("---")
		print(StG)
		print(StA)

		# F1-score of 1 if both spans are empty (AND)
		if StG == [] and StA == []: 
			F1_scores.append(1)
		# F1-score of 0 if only one of the spans is empty (XOR)
		elif StG == [] or StA == [] and StG != StA: 
			F1_scores.append(0)
		# Calculate F1-score if neither of the spans is empty (NOR)
		else:
			intersection = list(set(StA) & set(StG))

			# Calculate precision (P) and recall (R)
			P = len(intersection)/len(StA)
			R = len(intersection)/len(StG)

			# Catches ZeroDivisionError when P and R are 0
			try:
				F1 = (2*(P*R))/(P+R)
				F1_scores.append(F1)
			except:
				F1_scores.append(0)

	F1_score = mean(F1_scores)

	return F1_score


def main():

	SG, SA = get_spans("tsd_trial.csv", "results.csv")

	F1_score = calculate_F1(SG, SA)
	print("System performance (avg. F1): {}".format(round(F1_score, 3)))


if __name__ == '__main__':
	main()
