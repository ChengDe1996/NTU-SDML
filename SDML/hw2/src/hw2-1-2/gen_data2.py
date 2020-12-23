import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from multiprocessing import Pool
from os.path import join
import random
start_time = time.time()


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help = 'training_data')
	parser.add_argument('--output_dir', help = 'output_directory')
	parser.add_argument('--mode', help = 'how many word control', type = int, default = 1 )
	parser.add_argument('--new_data', help = 'new data', default = 'new_data2.txt')


	args = parser.parse_args()
	return args


def build_training_data(in_path, out_path):
	with open(in_path, "r") as infile:
		indata = infile.readlines()
	sos = '<SOS>'
	eos = '<EOS>'
	split_word_sen = []
	init = list(indata[0])
	init.pop()
	init.insert(0,sos)
	init.append(eos)
	split_word_sen.append(list(init))
	ctr_list = []
	for i in range(1,len(indata)):
		word_list = list(indata[i])
		word_list.pop()
		if random.random() > 0.4:
			ctr_num1 = random.randint(1,len(word_list))
			ctr_num2 = random.randint(1,len(word_list))
			if(ctr_num1>ctr_num2):
				temp = ctr_num1
				ctr_num1 = ctr_num2
				ctr_num2 = temp
			ctr_tar1 = word_list[ctr_num1-1]
			ctr_tar2 = word_list[ctr_num2-1]
		else:
			ctr_num1 = random.randint(1,len(word_list))
			ctr_num2 = ctr_num1
			ctr_tar1 = word_list[ctr_num1-1]
		temp = [ctr_num1, ctr_num2]
		ctr_list.append(temp)
		word_list.insert(0,sos)
		word_list.append(eos)
		if ctr_num1 == ctr_num2:
			split_word_sen[i-1].append(str(ctr_num1))
			split_word_sen[i-1].append(ctr_tar1)
		else:

			split_word_sen[i-1].append(str(ctr_num1))
			split_word_sen[i-1].append(ctr_tar1)
			split_word_sen[i-1].append(str(ctr_num2))
			split_word_sen[i-1].append(ctr_tar2)

		split_word_sen.append(word_list)

	with open(out_path,"w") as outfile:
		for i in range(len(split_word_sen)):
			for j in range(len(split_word_sen[i])):
				outfile.write(split_word_sen[i][j])
				outfile.write(' ')

			outfile.write('\n')

	return ctr_list



def main():
	args = parse_args()

	ctr_list = build_training_data(args.input, args.new_data)
	ctr_list = np.array(ctr_list)
	print(np.shape(ctr_list))
	np.save('ctr_list.npy', ctr_list)



if __name__ == '__main__':
	main()
