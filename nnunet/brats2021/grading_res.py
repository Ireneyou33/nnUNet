import os
import numpy as np
import pandas as pd

from nnunet.brats2021.path import nnunet_path

grading_file = os.path.join(nnunet_path, "brats2021", "grading_res.csv")

if os.path.isfile(grading_file):
    df_grad = pd.read_csv(grading_file, index_col='name')


def get_grad(p):
    if df_grad is not None:
        grading_pvalue = df_grad.loc[p]['grading_pvalue']
        LGG, HGG = grading_pvalue[1:-1].split()
        if np.isclose(float(LGG), 1):
            return "LGG"
        else:
            return "HGG"
    else:
        return None
