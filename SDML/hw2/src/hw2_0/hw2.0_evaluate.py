import pandas as pd 
import numpy as np
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--predict_file', type=str, required=True)
parser.add_argument('--true_file', type=str, required=True)
args = parser.parse_args()

def evaluate(predict_file,true_file):
	with open(predict_file,'rb') as f_i:
		predict = f_i.readlines()
	with open(true_file,'rb') as f_o:
		true = f_o.readlines()
	
	correct = 0
	assert len(predict) == len(true), "Predict file and true file should have same length."
	for i in range(len(predict)):
		if predict[i] == true[i]:
			correct += 1
		#else:
			#print(predict[i],true[i])
			#time.sleep(1)
	return (correct/len(predict))

print (evaluate(args.predict_file,args.true_file))