import os.path

from nnunet.dataset_conversion.Task501_BraTS_2021 import evaluate_BraTS_folder
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--pred-folder', help="模型预测结果", required=False)
    parser.add_argument('-g', "--gt-folder", help="真实结果", required=False)
    parser.add_argument('-r', "--result-file", help="真实结果", required=False, default=None)

    args = parser.parse_args()

    data_dir_pred = args.pred_folder
    data_dir_gt = args.gt_folder

    root_folder = "/mnt/ngshare/PersonData/qingyu/dataset/nnUNet/RESULTS_FOLDER/Predict_Result/Predict_Train_Post"
    sub_folders = os.listdir(root_folder)

    for sub_folder in sub_folders:
        if "Post750" not in sub_folder:
            continue
        data_dir_pred = os.path.join(root_folder, sub_folder, "Out")

        if args.result_file is None:
            result_file = os.path.join(data_dir_pred, "results.csv")
        else:
            result_file = args.result_file

        print(f'data_dir_pred : {data_dir_pred}')
        print(f'data_dir_gt   : {data_dir_gt}')
        print(f'result-file   : {result_file}')

        # 计算评价指标的结果文件
        if not os.path.isfile(result_file):
            evaluate_BraTS_folder(data_dir_pred, data_dir_gt, num_processes=6, out_file=result_file)
