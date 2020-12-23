import pandas as pd
from tqdm import tqdm
import numpy as np

from output_csv import output_csv

def pred_to_output_format(pred, thres=0.5, output_raw=False):
    cols = ['THEORETICAL', 'ENGINEERING', 'EMPIRICAL', 'OTHERS']
    output_df = pd.DataFrame(columns=cols)
    for row in tqdm(pred, desc='transfer pred to output format'):
        output_df = output_df.append({
            elm[0]: elm[1]
            for elm in row
        }, ignore_index=True)
    if output_raw:
        return output_df.values
    output_df = 0 + (output_df > 0.5)
    return output_df.values

def main():
    import json
    pred = json.load(open('./outputs/2_xlnet_epoch-10_0.664.json'))
    pred = pred_to_output_format(pred)

    pred_pad = np.zeros((40000, pred.shape[1]), int)
    pred_pad[:20000] = pred
    pred_fn = './outputs/2.csv'
    output_csv(pred_pad, pred_fn)
    print('Outputed to ', pred_fn)


if __name__ == '__main__':
    main()
