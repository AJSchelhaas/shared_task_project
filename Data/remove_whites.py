import sys

with open('Silver/silver_data_1.csv') as file:
	outfile = open('Silver/new_silver.csv', 'w')


	for line in file:
		if len(line)<=2:
			print(line)
		if '\n' in line:
			pass
		else:
			outfile.write(line)