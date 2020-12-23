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
	parser.add_argument('--input', help = 'training_data', default = 'new_data2.txt')
	parser.add_argument('--input2', help = 'target_data', default = 'new_target.txt')
	parser.add_argument('--output_dir', help = 'output_directory', default='pre_out2')
	parser.add_argument('--mode', help = 'how many word control', type = int, default = 1 )
	parser.add_argument('--new_data', help = 'new data', default = 'new_data.txt')


	args = parser.parse_args()
	return args


def load_data(path):
	"""load data from train.txt"""
	with open(path) as infile:
		dataset = infile.readlines()
	'''
	for i in tqdm(range(len(dataset))):
		dataset[i] = dataset[i][5:]
		dataset[i] = dataset[i][:-6]
	'''
	return dataset
def dele_long(source, target, ctr_list):
	source_split = []
	target_split = []
	ctr_list = list(ctr_list)
	#print(len(source), len(target))
	for i in range(len(source)):
		word_list = source[i].split()
		if (len(word_list)>35):
			word_list = word_list[:30]+word_list[-5:]
		source_split.append(word_list)
	for i in range(len(target)):
		word_list = target[i].split()
		if(len(word_list)>35):
			word_list = word_list[:30]+ word_list[-5:]
			#print(word_list)
		target_split.append(word_list)
	#print(len(source_split), len(target_split))
		#if(len(word_list)>35):
		#	idx_list.append(i)
		#	idx_list.append(i-1)
	#idx_list.sort(reverse = True)
	#for idx in idx_list:
	#	source.pop(idx)
	#	target.pop(idx)
	#	ctr_list.pop(idx)

	#print(np.shape(source), np.shape(target), np.shape(ctr_list))

	return source_split, target_split, ctr_list

def data_pair(source_dataset, target_dataset):
	source = []
	target = []
	for i in range(len(source_dataset)-1):
		source.append(source_dataset[i])
		target.append(target_dataset[i+1])
	return source, target


def collect_words(dataset, source_split, target_split):
	"""get word set"""
	split_word_sen = []
	print(len(source_split), len(target_split))
	words = set()
	for i in range(len(dataset)):
		temp = dataset[i].split()
		split_word_sen.append(temp)
		words.update(temp)

	for i in range(len(source_split)):
		#source_temp = source[i].split()
		#target_temp = target[i].split()
		#source_word_list = list(source[i])
		#target_word_list = list(target[i])
		#source_split.append(source_temp)
		#target_split.append(target_temp)
		words.update(source_split[i])
		words.update(target_split[i])
	print(len(source_split[1]),source_split[1])
	print(target_split[10])

	return words, split_word_sen, source_split, target_split

def load_glove(path):
	"""load glove word embedding"""
	print('[Time {}] loading GLOVE.'.format(time.time() - start_time))
	glove = {}
	with open(path, encoding='utf8') as f:
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			glove[word] = coefs
	print('[Time {}] Finish loading GLOVE.'.format(time.time() - start_time))
	return glove

def sentence_to_indices(sentence, word_dict):
	"""transform words in a sentence to indices"""
	return [word_dict.to_index(word) for word in list(sentence)]

def convert_to_indices(sen_list, word_dict):
	"""transfer whole dataset"""
	ind_sent_list = []
	for sentence in tqdm(sen_list):
		ind_sent_list.append(sentence_to_indices(sentence, word_dict))
	return ind_sent_list


def calc_sen_len(source_list, target_list):
	"""generate a list of len of whole sentence"""
	len_of_source_list = []
	len_of_target_list = []
	for sentence in tqdm(source_list):
		len_of_source_list.append(len(sentence.split()))
	for sentence in tqdm(target_list):
		len_of_target_list.append(len(sentence.split()))

	i = len_of_source_list.index(max(len_of_source_list))
	print(i)
	print(source_list[i], target_list[i])
	print(source_list[i+1], target_list[i+1])
	print(source_list[i-1], target_list[i-1])
	print(max(len_of_source_list),max(len_of_target_list))
	return len_of_source_list, len_of_target_list

def gen_idx2words_matrix(word_sets):
	""" Generate idx2word & embedding matrix from the given dictionary. """
	print('len:', len(word_sets))
	index2word = ['<PAD>','<UNK>','<SOS>','<EOS>']
	word2index = {index2word[0]:0, 
				  index2word[1]:1, 
				  index2word[2]:2, 
				  index2word[3]:3}
	for word in index2word:
		word_sets.discard(word)

	word_dim = 300
	pad_vec = np.zeros(word_dim)
	unk_vec = np.ones(word_dim)
	sos_vec = np.random.rand(word_dim)*2-1
	eos_vec = np.random.rand(word_dim)*2-1
	look_up_matrix = [pad_vec, unk_vec, sos_vec, eos_vec]
	for word in tqdm(word_sets):
		word2index[word] = len(index2word)
		index2word.append(word)

	return index2word, word2index






def convert_n_pad_token(split_word_sen, word2index, pad_len):
	""" transfer data from words to indices with padding"""
	ind_sent_list = np.zeros((len(split_word_sen),pad_len),int)
	for i, sentence in enumerate(split_word_sen):
		word_list = [word2index.get(word, 1) for word in sentence]
		ind_sent_list[i, :len(word_list)] = word_list

	return ind_sent_list







def main():
	args = parse_args()

	source_dataset = load_data(args.input)
	target_dataset = load_data(args.input2)
	ctr_list = np.load('ctr_list.npy')
	#print(np.shape(source_dataset), np.shape(target_dataset))
	source, target = data_pair(source_dataset, target_dataset)
	source_split, target_split, ctr_list = dele_long(source, target, ctr_list)
	ctr_list = np.array(ctr_list)
	#print(len(source_split), len(target_split))
	word_sets, split_word_sen, source_split, target_split= collect_words(source_dataset, source_split, target_split)
	#print(len(word_sets))
	#word_dict = load_glove('glove.6B.300d.txt')
	index2word, word2index = gen_idx2words_matrix(word_sets)
	print(len(index2word), len(word2index))
	#look_up_matrix = np.array(look_up_matrix)
	#source_sen_len, target_sen_len = calc_sen_len(source_, target)
	#max_pad_len = 35
	#print('pad_len : ', max_pad_len)
	#print(source_split[0])
	#print(target_split[0])
	with open('source_doc.txt', "w") as outfile:
		for i in range(len(source_split)):
			for j in range(len(source_split[i])):
				outfile.write(source_split[i][j])
				outfile.write(' ')
			outfile.write('\n')
	max_pad_len = 35
	source_list = convert_n_pad_token(source_split, word2index, max_pad_len)
	target_list = convert_n_pad_token(target_split, word2index, max_pad_len)
	print('[Time {}] Finish converting.'.format(time.time() - start_time))

	#output
	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)

	with open(join(args.output_dir, 'index2word.json'), 'w') as outfile:
		json.dump(index2word, outfile)
	with open(join(args.output_dir, 'word2index.json'), 'w') as outfile:
		json.dump(word2index, outfile)

	#np.savetxt(join(args.output_dir, 'look_up_matrix.csv'), look_up_matrix, delimiter=',')
	print(np.shape(source_list))
	print(np.shape(target_list))
	print(np.shape(ctr_list))

	np.save(join(args.output_dir,'source_list.npy'), source_list)
	np.save(join(args.output_dir,'target_list.npy'), target_list)
	np.save(join(args.output_dir,'ctr_list.npy'), ctr_list)
	#with open(join(args.output_dir, 'source_sent_len.json'), 'w') as outfile:
	#	json.dump(source_sen_len,outfile)

	#with open(join(args.output_dir, 'target_sent_len.json'), 'w') as outfile:
	#	json.dump(target_sen_len,outfile)

if __name__ == '__main__':
	main()