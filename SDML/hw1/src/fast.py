import argparse
import json
import logging
import os
import pickle

from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy, F1
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from metric import F1 as myF1
from output_csv import output_csv
from pred_to_output_format import pred_to_output_format

COLS = ['THEORETICAL', 'ENGINEERING', 'EMPIRICAL', 'OTHERS']

def parse_args():
    """ Parse args. """
    parser = argparse.ArgumentParser()

    # output
    parser.add_argument('model_idx', type=int,  help='model index')

    # model
    parser.add_argument('--raw_pred', help='raw output prediction')

    # toggle True / False
    parser.add_argument('--is_large', action='store_true',
                        help='True for is_large (False for is_base)')
    parser.add_argument('--disable_amp', action='store_true',
                        help='True for disabling amp')
    parser.add_argument('--no_tune', action='store_true',
                        help='Don\'t tune the params of BERT.')
    parser.add_argument('--pos_weight', action='store_true',
                        help='True for use pos weight.')

    # numbers
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--lr', '--learning_rate', type=float, default=5e-5,
                        help='The initial learning rate for Adam')
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('-s', '--seed', type=int, default=6174,
                        help='random seed')
    parser.add_argument('-e', '--epoch_size', type=int, default=10,
                        help='epoch size')
    parser.add_argument('--valid', type=float, default=0.001,
                        help='Split validation data to observe loss & f1.')
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('-t', '--thres', type=float, default=0.5,
                        help='threshold')

    args = parser.parse_args()
    return args

def learner_pred(learner, test_texts, thres=0.5, batch_num=100):
    pred = []
    for batch_idx in range(len(test_texts) // batch_num):
        text = test_texts[batch_idx * batch_num: (batch_idx + 1) * batch_num]
        pred += learner.predict_batch(text)

    output_df = pd.DataFrame(columns=COLS)
    for row in tqdm(pred, desc='transfer pred to output format'):
        output_df = output_df.append({
            elm[0]: elm[1]
            for elm in row
        }, ignore_index=True)
    output_df = 0 + (output_df > thres)
    return output_df.values

def run():
    args = parse_args()

    with open('fast_bert_val.csv') as infile:
        val_df = pd.read_csv(infile)
        val_text = val_df['text'].tolist()
        val_label = torch.Tensor(val_df[COLS].values).float()

    DATA_PATH = './'
    LABEL_PATH = './'
    OUTPUT_DIR = './'
    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger()
    databunch = BertDataBunch(DATA_PATH,
                              LABEL_PATH,
                              tokenizer='xlnet-base-cased',
                              train_file='fast_bert_train.csv',
                              val_file='fast_bert_val.csv',
                              label_file='labels.csv',
                              text_col='text',
                              label_col=['THEORETICAL', 'ENGINEERING', 'EMPIRICAL', 'OTHERS'],
                              batch_size_per_gpu=8,
                              max_seq_length=512,
                              multi_gpu=True,
                              multi_label=True,
                              model_type='xlnet')

    device_cuda = torch.device('cuda')
    metrics = [{'name': 'f1', 'function': F1}]
    learner = BertLearner.from_pretrained_model(databunch,
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

    # train
    learner.fit(epochs=args.epoch_size,
                lr=args.lr,
                validate=True,
                schedule_type='warmup_cosine',
                optimizer_type='lamb')
    learner.save_model()

    os.rename('./model_out', 'model_out_{}'.format(args.model_idx))

    # test
    with open('./test_abstract.json') as infile:
        test_texts = json.load(infile)
    test_pred = learner_pred(learner, test_texts)

    pred_pad = np.zeros((40000, test_pred.shape[1]), int)
    pred_pad[:20000] = test_pred
    pred_fn = './{}.csv'.format(args.model_idx)
    output_csv(test_pred, pred_fn)
    print('Outputed to ', pred_fn)

    # get f1
    val_pred = learner_pred(learner, val_text)
    f1_score = myF1()
    f1_score.update(torch.Tensor(val_pred), val_label)
    print('valid f1_score =', f1_score.get_score())

if __name__ == '__main__':
    run()
