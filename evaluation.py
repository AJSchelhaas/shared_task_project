#!/urs/bin/python3 
import pandas as pd
from sklearn.metrics import f1_score
import re
import json
from sklearn.preprocessing import MultiLabelBinarizer

source_data = pd.read_csv('tsd_train.csv')
target_data = open('results.csv','r')
multibinarizer = MultiLabelBinarizer()

target_spans = []
for line in target_data:
	target_span = []
	try:
		span = re.search('"\[(.*?)\]"', line).group(1)
		
		span = '['+span+']'
		print(span)
		target_spans.append(span)
	except:
		pass

	#target_spans.append(target_span)
print(len(target_spans))

#print(len(target_spans))
#print(len(source_data['spans']))
#for y_true, y_pred in zip(source_data['spans'], target_data['spans']):
for y_true, y_pred in zip(source_data['spans'], target_spans):
	y_true = json.loads(y_true)
	y_true = multi.fit(y_true).transform(A)

	y_pred = json.loads(y_pred)
	pred = multi.fit(y_true).transform(A)
	f1 = f1_score(y_true, y_pred)
	print(f1)

	print('true', y_true, 'pred', y_pred)
#	print(y_true)
#	print(y_pred)

	#f1 = f1_score(y)	 