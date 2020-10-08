import pyconll
import math

sent_list = []

file_to_split = 'Data/converted_data.conll'
conll_struc = pyconll.load_from_file(file_to_split)
struc_len = len(conll_struc)

split_index_1 = int(0)
split_index_2 = int(0.1 * struc_len)
split_index_3 = int(0.3 * struc_len)
split_index_4 = int(struc_len)

sent_list.append(conll_struc[split_index_1:split_index_2])
sent_list.append(conll_struc[split_index_2:split_index_3])
sent_list.append(conll_struc[split_index_3:split_index_4])

ext_list = ['dev', 'test', 'train']
for i in range(3):
	ext_str = ext_list[i]
	file_name = file_to_split[:-6]
	new_file_name = file_name + "_" + ext_str + ".conll"
	conll_string = sent_list[i].conll()

	with open(new_file_name, "w") as f:
		f.writelines(conll_string)