#!/usr/bin/python3

import csv
from nltk.tokenize import word_tokenize
import nltk
import spacy
import re
def read_corpus():
	nlp = spacy.load("en_core_web_sm")
	outfile = open('Silver/conv_sdata_annot.conll', 'w')
	with open('Silver/silver_data_1.csv') as data:
		file = csv.reader(data)
		next(file, None)
		sent_id = 1
		for row in file:
			#print(row)
			#print(sent_id)
			if len(row)<2:
				pass
			else:
				try:
					#print('test')
					#print(len(row))
					text = row[1].replace('\n',' ')
					span_string = row[0].strip('][').split(', ')
					if span_string == ['']:
						span =[]
					else:
						span = [int(e) for e in span_string]
			
					if span == []:
						toxic_words = text
					else:
						toxic_word = []
						prev_index = span[0]-1
						for index in span:
							if index == prev_index+1:
								toxic_word.append(text[index])
								prev_index = index
							else:
								toxic_word.append(' ')
								toxic_word.append(text[index])
								prev_index = index
						
						toxic_word = ''.join(toxic_word)
						toxic_words = toxic_word.split(' ')
				
					doc = nlp(text)
					#print(len)
					#print(doc)
					i = 0
					u = '_'
					for token in doc:
						if token.text in toxic_words:
							label = 1
						else:
							label = 0
						outfile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(str(i+1), token.text,token.lemma_,token.pos_,token.tag_,u,u,token.dep_,u,label))
						i += 1
							#print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_)

					#tokens = word_tokenize(text)
					#pos_tags = nltk.pos_tag(tokens)
					# outfile.write('#sent_id = '+str(sent_id)+'\n')
					# outfile.write('#text = '+text+'\n')
					# outfile.write('#span = '+row[0]+'\n')
					# outfile.write('#toxic words = '+str(toxic_words)+'\n')
					#sent_id+=1
					#u = '_'
					#for i in range(len(pos_tags)):
					#	pos_token = pos_tags[i]
					#	if pos_token[0] in toxic_words:
					#		label = 1
							# outfile.write(str(i+1)+'\t'+pos_token[0]+'\t'+'_'+'\t'+pos_token[1]+'\t'+'_'+'\t'+'_'+'\t'+'_'+'\t'+'_'+'\t'+'_'+'\t'+'tox=1'+'\n')
					#	else:
					#		label = 0
					#		# outfile.write(str(i+1)+'\t'+pos_token[0]+'\t'+'_'+'\t'+pos_token[1]+'\t'+'_'+'\t'+'_'+'\t'+'_'+'\t'+'_'+'\t'+'_'+'\t'+'tox=0'+'\n')
					#	outfile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(str(i+1), pos_token[0],u,pos_token[1],u,u,u,u,u,label))
					sent_id += 1
					outfile.write('\n')
				except:
					pass
			#else:
			#	pass

def main():

	read_corpus()

if __name__ == '__main__':
	main()