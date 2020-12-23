import numpy as np
import pandas as pd

def output_csv(pred, pred_fname):
    """ Output to csv. """
    cols = ['THEORETICAL', 'ENGINEERING', 'EMPIRICAL', 'OTHERS']

    # create df
    pred_df = pd.DataFrame({
        'order_id': ['T{:05d}'.format(i) for i in range(1, pred.shape[0] + 1)],
        cols[0]: pred[:, 0],
        cols[1]: pred[:, 1],
        cols[2]: pred[:, 2],
        cols[3]: pred[:, 3],
    })

    # output
    pred_df[['order_id',
             cols[0],
             cols[1],
             cols[2],
             cols[3]]].to_csv(pred_fname, index=False)
