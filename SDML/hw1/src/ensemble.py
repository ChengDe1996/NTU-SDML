import json

import numpy as np
import pandas as pd
import torch

from metric import F1
from pred_to_output_format import pred_to_output_format
from output_csv import output_csv

COLS = ['THEORETICAL', 'ENGINEERING', 'EMPIRICAL', 'OTHERS']

def calc_f1(vali, label):
    f1 = F1()
    f1.update(vali, label)
    return f1.get_score()

def run():
    # pred
    ratio = np.array([0.2, 0.2, 0.6]).reshape(3, 1, 1)
    thres = 0.3

    # test
    pd_vali_path = './1.csv'
    json_path = [
        './bert_test.json',
        './xlnet_test.json',
    ]

    vali = []
    my_vali = pd.read_csv(pd_vali_path).values[:, 1:]
    my_vali = np.array(my_vali, float)
    vali.append(my_vali)
    for j in json_path:
        p = json.load(open(j))
        pred = pred_to_output_format(p, output_raw=True)
        vali.append(pred)
    vali = np.array(vali)
    pred_pad = np.zeros((3, 40000, 4))
    pred_pad[:, :20000] = vali
    vali = pred_pad

    vali_en = 0 + ((vali * ratio.reshape(3, 1, 1)).sum(0) > thres)
    output_csv(vali_en, './final_output.csv')

if __name__ == '__main__':
    run()
