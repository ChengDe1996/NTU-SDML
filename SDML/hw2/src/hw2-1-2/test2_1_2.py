import numpy as np
import json
import pandas as pd
import argparse
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import Encoder, Decoder, Seq2Seq
from os.path import join
from tqdm import tqdm
import os

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mdir',help = 'modle_directory', default = 'model_out2')
	parser.add_argument('--source_list', help = 'source_list', default = join('pre_out2', 'source_list.npy'))
	parser.add_argument('--target_list', help = 'target_list', default = join('pre_out2', 'target_list.npy'))
	parser.add_argument('--ctr_list', help = 'ctr_list', default = join('pre_out2', 'ctr_list.npy'))
	parser.add_argument('--index2word', help = 'index2word', default = join('pre_out2','index2word.json'))
	parser.add_argument('--word2index', help = 'word2index', default = join('pre_out2', 'word2index.json'))
	#parser.add_argument('--look_up_matrix', help = 'look_up_matrix', default = join('pre_out', 'look_up_matrix.csv'))
	parser.add_argument('--source_len', help = 'source_len', default = join('pre_out2', 'source_sent_len.json'))
	parser.add_argument('--target_len', help = 'target_len', default = join('pre_out2', 'target_sent_len.json'))
	parser.add_argument('--epoch_size', type = int, default = 20 )
	parser.add_argument('--batch_size', type = int, default = 128)
	parser.add_argument('--clip', type = float, default = 5.0)
	parser.add_argument('--layers', type = int, default = 2)
	parser.add_argument('--dropout', type = float, default = 0.2)
	parser.add_argument('--hidden_dim', type = int, default = 128)
	parser.add_argument('--path1', help = 'pkl_path1', default = join('model_out2','model_e-7.pkl'))
	#parser.add_argument('--path2', help = 'pkl_path2', default = join('model_out2','encoder_e-0.pkl'))
	#parser.add_argument('--path3', help = 'pkl_path3', default = join('model_out2','decoder_e-0.pkl'))

	args = parser.parse_args()
	return args


def test(model, optim, loss_func, dataloader, index2word, word2index):
	model.eval()

	predict_sen = []
	source_sen = []
	with torch.no_grad():
		for _, data  in tqdm(enumerate(dataloader), total = len(dataloader)):
			source, answer = data
			optim.zero_grad()
			output = model(source, answer, teacher_forcing = 0.0)
			predict = output.detach().argmax(dim = -1)
			predict[:,0] = word2index['<SOS>']

			for j in range(predict.shape[0]):
				predict_sen.append([index2word[k] for k in predict[j]])
			for j in range(len(source)):
				source_sen.append([index2word[k] for k in source[j]])

		return predict_sen, source_sen

def prepare_data(source_list, target_list, ctr_list, bs, device):
	#look_up_matrix = torch.from_numpy(look_up_matrix).float().to(device)
	#(train, valid) = train_test_split(sen_list, test_size = 0.1, random_state = 1234)
	(source, target) = (torch.from_numpy(source_list).to(device), torch.from_numpy(target_list).to(device))
	testing_set= TensorDataset(source, source)
	testing_data_loader= DataLoader(testing_set, batch_size = bs )

	return testing_data_loader


def main():

	args = parse_args()
	random.seed(1234)
	torch.manual_seed(1234)
	torch.backends.cudnn.deterministic = True
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

	source_list = np.load(args.source_list)
	target_list = np.load(args.target_list)
	ctr_list = np.load(args.ctr_list)
	index2word = json.load(open(args.index2word))
	word2index = json.load(open(args.word2index))
	print(len(index2word), len(word2index))

	test_data_loader= prepare_data(source_list, target_list, ctr_list, args.batch_size, device)

	hyper_param = {'out_dim':len(index2word),
					'embed_dim': 300,
					'hidden_dim': args.hidden_dim,
					'layers': args.layers,
					'dropout': args.dropout,
					'clip':args.clip,
					}



	encoder = Encoder(hyper_param['embed_dim'], hyper_param['hidden_dim'], hyper_param['layers'], hyper_param['dropout'] ).to(device)
	decoder = Decoder(hyper_param['embed_dim'], hyper_param['hidden_dim'], hyper_param['layers'], hyper_param['dropout'], hyper_param['out_dim']).to(device)
	model = Seq2Seq(hyper_param['embed_dim'],encoder, decoder, hyper_param['out_dim'], device).to(device)
	model.load_state_dict(torch.load(args.path1))
	optimizer = optim.Adam(model.parameters())
	loss_func = nn.CrossEntropyLoss(ignore_index = 0)
	predict_list, source_sen = test(model, optimizer, loss_func, test_data_loader, index2word, word2index)

	print(np.shape(predict_list))
	with open('predict_doc.txt',"w") as outfile:
		for i in range(len(predict_list)):
			for j in range(len(predict_list[i])):
				outfile.write(predict_list[i][j])
				if(predict_list[i][j] == '<EOS>'):
					break
				outfile.write(' ')

			outfile.write('\n')

	with open('source_doc.txt',"w") as outfile:
		for i in range(len(source_sen)):
			temp = 0
			for j in range(len(source_sen[i])):
				if (temp == 1 and (source_sen[i][j] == '<EOS>' or source_sen[i][j] == '<PAD>' or source_sen[i][j] == '<pad>')):
					break
				outfile.write(source_sen[i][j])
				outfile.write(' ')

				if(temp == 0 and source_sen[i][j] == '<EOS>'):
					temp = 1

			outfile.write('\n')



if __name__ =='__main__':
	main()


