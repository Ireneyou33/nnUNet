import argparse
import pandas as pd
import SimpleITK as sitk

from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.brats2021.path import nnunet_path

current_path = os.path.join(nnunet_path, "brats2021")


if __name__ == "__main__":
    """
    THIS CODE IS A MESS. IT IS PROVIDED AS IS WITH NO GUARANTEES. YOU HAVE TO DIG THROUGH IT YOURSELF. GOOD LUCK ;-)

    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--pred-folder', help="模型预测结果", required=True)
    parser.add_argument("-r", '--result-file', help="模型预测结果", required=True)

    args = parser.parse_args()

    data_dir_pred = args.pred_folder
    result_file = args.result_file

    files = subfiles(data_dir_pred, suffix='.nii.gz', join=True)

    count = []
    for f in files:
        arr_pred = sitk.GetArrayFromImage(sitk.ReadImage(f))
        label1 = (arr_pred == 1).sum()
        label2 = (arr_pred == 2).sum()
        label3 = (arr_pred == 3).sum()
        filename = os.path.basename(f)
        count.append([filename, label1, label2, label3])

    df = pd.DataFrame(count, columns=["id", "label_1", "label_2", "label_3"])
    pred_folder_name = os.path.basename(data_dir_pred)
    df.to_csv(result_file, index=False)
