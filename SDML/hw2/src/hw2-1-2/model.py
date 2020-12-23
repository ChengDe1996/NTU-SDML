import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
	"""simple emcoder"""
	def __init__(self, embed_dim, hidden_dim, layers, dropout):
		"""initialization"""
		super().__init__()

		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.layers = layers
		self.dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(embed_dim, hidden_dim, layers, dropout = dropout, batch_first = True)

	def forward(self, embed):
		embed = self.dropout(embed)
		output = embed
		output, hidden = self.gru(output)

		return hidden

class Decoder(nn.Module):
	"""simple decoder"""
	def __init__(self, embed_dim, hidden_dim, layers, dropout, out_dim):
		super().__init__()

		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.layers = layers
		self.dropout = dropout
		self.out_dim = out_dim
		self.dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(embed_dim, hidden_dim, layers, dropout = dropout, batch_first = True)
		self.out = nn.Linear(hidden_dim, out_dim)
		self.softmax = nn.LogSoftmax(dim = 1)

	def forward(self, embed, hidden):
		embed = self.dropout(embed)
		output = embed
		output, hidden = self.gru(output, hidden)
		predict = self.softmax(self.out(output.squeeze(1)))

		return predict, hidden

class Seq2Seq(nn.Module):
	"""seq2seq model"""
	def __init__(self, embed_dim, encoder, decoder, out_dim, device, pretrained_embd= None,):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.out_dim = out_dim
		self.device = device
		self.embedding = nn.Embedding(out_dim, embed_dim)
		#self.embedding.weight = nn.Parameter(pretrained_embd)
		self.embedding.requires_grad = True

	def forward(self, source, answer, teacher_forcing = 0.5):
		batch_size = source.shape[0]
		sen_len = source.shape[1]
		outputs_prob = torch.zeros(batch_size, sen_len, self.out_dim).to(self.device)
		encoder_embed = self.embedding(source)
		hidden_state = self.encoder(encoder_embed)
		inputs = answer[:, 0]
		for i in range(1, sen_len):
			inputs = inputs.unsqueeze(1)
			decoder_embed = self.embedding(inputs)
			output, hidden_state = self.decoder(decoder_embed, hidden_state)
			outputs_prob[:, i] = output.squeeze(1)
			yes_teacher =  random.random() < teacher_forcing
			predict = output.argmax(1)
			if yes_teacher:
				inputs = answer[:, i]
			else:
				inputs = predict
		return outputs_prob













