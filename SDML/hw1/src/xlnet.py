#!/usr/bin/env python
from fast_bert.data_cls import BertDataBunch
from fast_bert.prediction import BertClassificationPredictor
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy, F1
import logging, torch, os, pickle, json
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'
def train():
	DATA_PATH = "./"
	LABEL_PATH = "./"
	OUTPUT_DIR = "./model/xlnet-base"
	logging.basicConfig(level=logging.NOTSET)
	logger = logging.getLogger()
	databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
				tokenizer='xlnet-base-cased',
				train_file='fast_bert_train.csv',
				val_file='fast_bert_val.csv',
				label_file='labels.csv',
				text_col='text',
				label_col=["THEORETICAL", "ENGINEERING", "EMPIRICAL", "OTHERS"],
				batch_size_per_gpu=2,
				max_seq_length=512,
				multi_gpu=True,
				multi_label=True,
				model_type='xlnet')
	device_cuda = torch.device("cuda")
	metrics = [{'name': 'f1', 'function': F1}]
	learner = BertLearner.from_pretrained_model(
					databunch,
					pretrained_path='xlnet-base-cased',
					metrics=metrics,
					device=device_cuda,
					logger=logger,
					output_dir=OUTPUT_DIR,
					finetuned_wgts_path=None,
					warmup_steps=500,
					multi_gpu=True,
					is_fp16=False,
					multi_label=True,
					logging_steps=50)
	learner.fit(epochs=10, lr=6e-5, validate=True, schedule_type="warmup_cosine", optimizer_type="lamb")

	learner.save_model()
	print("finish training")
	with open("test_abstract", "rb") as f:
		test_texts = pickle.load(f)
	val_texts = list(pd.read_csv("fast_bert_val.csv")["text"])
	batch_num = 100
	pred = []
	for batch_idx in range(len(val_texts) // batch_num):
		print(batch_idx)
		text = val_texts[batch_idx * batch_num: (batch_idx + 1) * batch_num]
		pred += learner.predict_batch(text)
	with open('./val_predictions', 'w') as outfile:
		json.dump(pred, outfile)
	pred = []
	for batch_idx in range(len(test_texts) // batch_num):
		text = test_texts[batch_idx * batch_num: (batch_idx + 1) * batch_num]
		print(batch_idx)
		pred += learner.predict_batch(text)
	with open('./xlnet_test.json', 'w') as outfile:
		json.dump(pred, outfile)

def test():
	MODEL_PATH='./model/xlnet-base/model_out'
	LABEL_PATH='./'
	predictor = BertClassificationPredictor(model_path=MODEL_PATH,
						label_path=LABEL_PATH,
						multi_label=True,
						model_type='xlnet',
						do_lower_case=True)
	with open("test_abstract", "rb") as f:
		test_texts = pickle.load(f)
	val_texts = list(pd.read_csv("fast_bert_val.csv")["text"])

	batch_num = 100
	pred = []
	for batch_idx in range(len(val_texts) // batch_num):
		print(batch_idx)
		text = val_texts[batch_idx * batch_num: (batch_idx + 1) * batch_num]
		pred += predictor.predict_batch(text)
	with open('./val_predictions', 'w') as outfile:
		json.dump(pred, outfile)
	pred = []
	for batch_idx in range(len(test_texts) // batch_num):
		text = test_texts[batch_idx * batch_num: (batch_idx + 1) * batch_num]
		print(batch_idx)
		pred += predictor.predict_batch(text)
	with open('./test_predictions', 'w') as outfile:
		json.dump(pred, outfile)

if __name__ == "__main__":
	train()	
	#test()
