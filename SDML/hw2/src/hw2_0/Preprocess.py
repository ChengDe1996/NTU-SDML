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
start_time = time.time()


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help = 'training_data', default = 'train.txt')
	parser.add_argument('--output_dir', help = 'output_directory', default = 'pre_out')


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

def collect_words(dataset):
	"""get word set"""
	split_word_sen = []

	words = set()
	for i in range(len(dataset)):
		word_list = dataset[i].split()
		split_word_sen.append(word_list)
		words.update(word_list)

	return words, split_word_sen

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
	return [word_dict.to_index(word) for word in sentence.split()]

def convert_to_indices(sen_list, word_dict):
	"""transfer whole dataset"""
	ind_sent_list = []
	for sentence in tqdm(sen_list):
		ind_sent_list.append(sentence_to_indices(sentence, word_dict))
	return ind_sent_list


def calc_sen_len(sen_list):
	"""generate a list of len of whole sentence"""
	len_of_sen_list = []
	for sentence in tqdm(sen_list):
		len_of_sen_list.append(len(sentence.split()))
	return len_of_sen_list

def gen_idx2words_matrix(word_sets, word_dict):
	""" Generate idx2word & embedding matrix from the given dictionary. """
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
		if word in word_dict:
			word2index[word] = len(index2word)
			index2word.append(word)
			embd = word_dict[word]
		else:
			embd = unk_vec
		look_up_matrix.append(embd)

	return index2word, word2index, look_up_matrix


def convert_n_pad_token(split_word_sen, word2index, pad_len):
	""" transfer data from words to indices with padding"""
	ind_sent_list = np.zeros((len(split_word_sen),pad_len),int)

	for i, sentence in enumerate(split_word_sen):
		word_list = [word2index.get(word, 1) for word in sentence]
		ind_sent_list[i, :len(word_list)] = word_list

	return ind_sent_list









def main():
	args = parse_args()
	dataset = load_data(args.input)
	word_sets, split_word_sen= collect_words(dataset)
	#print(len(word_sets))
	word_dict = load_glove('glove.6B.300d.txt')
	index2word, word2index, look_up_matrix = gen_idx2words_matrix(word_sets, word_dict)
	#print(word2index[0], word2index[1])
	look_up_matrix = np.array(look_up_matrix)
	sent_len = calc_sen_len(dataset)
	max_pad_len = max(sent_len)
	print('pad_len : ', max_pad_len)
	#print(split_word_sen[0])
	#print(split_word_sen[1])
	sen_list = convert_n_pad_token(split_word_sen, word2index, max_pad_len)
	#print(sen_list[0])
	#print(sen_list[1])
	print('[Time {}] Finish converting.'.format(time.time() - start_time))

	#output
	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)

	with open(join(args.output_dir, 'index2word.json'), 'w') as outfile:
		json.dump(index2word, outfile)
	with open(join(args.output_dir, 'word2index.json'), 'w') as outfile:
		json.dump(word2index, outfile)

	np.savetxt(join(args.output_dir, 'look_up_matrix.csv'), look_up_matrix, delimiter=',')
	np.save(join(args.output_dir,'sen_list.npy'), sen_list)

	with open(join(args.output_dir, 'sent_len.json'), 'w') as outfile:
		json.dump(sent_len,outfile)



if __name__ == '__main__':
	main()