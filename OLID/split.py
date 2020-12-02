
outfile = open('offensive_data.tsv', 'w')
with open('olid-training-v1.0.tsv') as old_data:
	for line in old_data:
		line = line.split('\t')
		if line[2] =='OFF':
			print('\t'.join(line))
			outfile.write(line[1]+'\n')
