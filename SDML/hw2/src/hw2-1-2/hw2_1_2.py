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

	args = parser.parse_args()
	return args

def prepare_data(source_list, target_list, ctr_list, bs, device):
	#look_up_matrix = torch.from_numpy(look_up_matrix).float().to(device)
	(train_source, valid_source, train_target, valid_target, train_ctr, valid_ctr) = train_test_split(source_list, target_list, ctr_list, test_size = 0.1, random_state = 1234)
	#print(train_targ)# valid_source, train_target, valid_target)
	(train_source, valid_source,
	train_target, valid_target,
	train_ctr, valid_ctr) = (torch.from_numpy(train_source).to(device), 
					  			   torch.from_numpy(valid_source).to(device),
					  			   torch.from_numpy(train_target).to(device),
					  			   torch.from_numpy(valid_target).to(device),
					  			   torch.from_numpy(train_ctr).to(device),
					  			   torch.from_numpy(valid_ctr).to(device))

	train_set, valid_set = (TensorDataset(train_source, train_target, train_ctr), TensorDataset(valid_source, valid_target, valid_ctr))
	train_data_loader, valid_data_loader = (DataLoader(train_set, batch_size = bs ), DataLoader(valid_set, batch_size = bs))

	return train_data_loader, valid_data_loader

def train_epoch(model, optim, loss_func, dataloader, i, index2word, word2index):
	model.train()
	epoch_loss = 0
	epoch_acc = 0
	#predict_sen = []

	for _, data  in tqdm(enumerate(dataloader), total = len(dataloader)):
		source, answer, ctr = data
		optim.zero_grad()
		output = model(source, answer)
		predict = output.detach().argmax(dim = -1)
		predict[:,0] = word2index['<SOS>']
		#print(np.shape(predict))
		#print([index2word[i] for i in source[0]]) 
		#print([index2word[i] for i in answer[0]])
		#print([index2word[i] for i in predict[0]])
		#if(i>10):
		#	print([index2word[k] for k in predict[10]])
		#	print([index2word[k] for k in answer.detach()[10]])

		#for j in range(predict.shape[0]):
		#	predict_sen.append([index2word[k] for k in predict[j]])


		print([index2word[i] for i in source[0]])
		print([index2word[i] for i in answer[0]])
		print([index2word[i] for i in predict[0]])
		#for j in range(predict.shape[0]):
			#acc = (predict[ctr[j]] == answer[ctr[j]])
			#epoch_acc += acc


		acc = ((answer.detach()[:, 1:] == predict[:, 1:])|(answer.detach()[:, 1:] == 0)&(answer.detach()[:, 1:] != 1)).all(dim = 1).float().mean().item()
		output, answer = (output[1:].view(-1, output.shape[-1]),answer[1:].view(-1))#why -1
		loss = loss_func(output, answer)
		loss.backward()

		optim.step()

		epoch_loss += loss.item()
		epoch_acc += acc

	#print([index2word[k] for k in output])
	#print([index2word[k] for k in answer])
	return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

def valid_epoch(model, optim, loss_func, dataloader, i, index2word, word2index):
	model.eval()
	epoch_loss = 0
	epoch_acc = 0
	#predict_sen = []
	for _, data in tqdm(enumerate(dataloader), total = len(dataloader)):
		source, answer, ctr = data
		output = model(source, answer)
		predict = output.detach().argmax(dim = -1)
		predict[:,0] = word2index['<SOS>']

		#for j in range(predict.shape[0]):
			#predict_sen.append([index2word[k] for k in predict[j]])
		#if(i>10):
		#	print([index2word[k] for k in predict[1]])
		#	print([index2word[k] for k in answer.detach()[1]])

		#print([index2word[i] for i in source[0]])
		#print([index2word[i] for i in answer[0]])
		#print([index2word[i] for i in predict[0]])

		acc = ((answer.detach()[:, 1:] == predict[:, 1:])|(answer.detach()[:, 1:] == 0)&(answer.detach()[:, 1:] != 1)).all(dim = 1).float().mean().item()
		output, answer = (output[1:].view(-1, output.shape[-1]),answer[1:].view(-1))
		loss = loss_func(output, answer)

		epoch_loss += loss.item()
		epoch_acc += acc
	return epoch_loss/len(dataloader), epoch_acc/len(dataloader)


def safe_epoch(model, encoder, decoder, optimizer, history, idx, mdir, hyper_param):
	if not os.path.exists(mdir):
		os.mkdir(mdir)

	with open(join(mdir,'hyper_param.json'), 'w') as outfile:
		json.dump(hyper_param, outfile, indent = 4)

	#with open(join(mdir, 'history.json'), 'w') as outfile:
		#json.dump(history, outfile, indent = 4)
	
	#checkpoint = {'model': model.state_dict()}
	#checkpoint2 = {'model':encoder.state_dict()}
	#checkpoint3 = {'model':decoder.state_dict()}
	torch.save(model.state_dict(), join(mdir,'model_e-{}.pkl'.format(idx)))
	#torch.save(encoder.state_dict(), join(mdir,'encoder_e-{}.pkl'.format(idx)))
	#torch.save(decoder.state_dict(), join(mdir,'decoder_e-{}.pkl'.format(idx)))





'''
def load(self, pkl_path):
    checkpoint = torch.load(pkl_path)
    self.model.load_state_dict(checkpoint['model'])
    self.opt.load_state_dict(checkpoint['optimizer'])
'''
def init_weights(model):
    for _, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def main():
	"""autoencoder"""
	args = parse_args()
	random.seed(1234)
	torch.manual_seed(1234)
	torch.backends.cudnn.deterministic = True
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
	#load data
	#sen_list = np.load(args.sen_list)
	source_list = np.load(args.source_list)
	target_list = np.load(args.target_list)
	print(len(source_list), len(target_list))
	ctr_list = np.load(args.ctr_list)
	index2word = json.load(open(args.index2word))
	word2index = json.load(open(args.word2index))
	print(len(index2word), len(word2index))

	#print(index2word[source_list[12][3]])
	#print([index2word[k] for k in source_list[0]])
	#print([index2word[k] for k in target_list[0]])
	#print([index2word[k] for k in source_list[99]])
	#print([index2word[k] for k in target_list[99]])
	#print([index2word[k] for k in source_list[1234]])
	#print([index2word[k] for k in target_list[1234]])

	#look_up_matrix = np.genfromtxt(args.look_up_matrix, delimiter=',')
	#source_len = json.load(open(args.source_len))
	#target_len = json.load(open(args.target_len))

	train_data_loader, valid_data_loader = prepare_data(source_list, target_list, ctr_list, args.batch_size, device)

	hyper_param = {'out_dim':len(index2word),
					'embed_dim': 300,
					'hidden_dim': args.hidden_dim,
					'layers': args.layers,
					'dropout': args.dropout,
					'clip':args.clip,
					}
	print(hyper_param['out_dim'])
	encoder = Encoder(hyper_param['embed_dim'], hyper_param['hidden_dim'], hyper_param['layers'], hyper_param['dropout'] ).to(device)
	decoder = Decoder(hyper_param['embed_dim'], hyper_param['hidden_dim'], hyper_param['layers'], hyper_param['dropout'], hyper_param['out_dim']).to(device)
	model = Seq2Seq(hyper_param['embed_dim'],encoder, decoder, hyper_param['out_dim'], device).to(device)
	model.apply(init_weights)
	#nn.init.uniform_(model.weight)
	"""should try"""
	optimizer = optim.Adam(model.parameters())
	loss_func = nn.CrossEntropyLoss(ignore_index = 0)
	""""""
	history = []
	for i in tqdm(range(args.epoch_size)):
		train_loss, train_acc = train_epoch(model, optimizer, loss_func, train_data_loader, i, index2word, word2index)
		valid_loss, valid_acc = valid_epoch(model, optimizer, loss_func, valid_data_loader, i, index2word, word2index)
		print({'train':{'loss': train_loss, 'acc':train_acc}, 'valid':{'loss':valid_loss,'acc':valid_acc},})
		history.append({'train':{'loss': train_loss, 'acc':train_acc}, 'valid':{'loss':valid_loss,'acc':valid_acc},})
		safe_epoch(model, encoder, decoder, optimizer, history, i, args.mdir, hyper_param)








if __name__ =='__main__':
	main()