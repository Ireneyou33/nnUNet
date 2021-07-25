import os.path

from nnunet.dataset_conversion.Task501_BraTS_2021 import evaluate_BraTS_folder


# def plot_evaluate


if __name__ == "__main__":
    data_dir_pred = "/home/anning/Dataset/ProjData/nnunet/RESULTS_FOLDER/nnUNet/3d_fullres/Task501_BraTS2021/nnUNetPlusPlusTrainerV2BraTS_Adam__nnUNetPlansv2.1/fold_0/validation_raw_best"
    data_dir_gt = "/home/anning/Dataset/ProjData/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task501_BraTS2021/labelsTr"

    print(f'data_dir_pred : {data_dir_pred}')
    print(f'data_dir_gt   : {data_dir_gt}')
    # 计算评价指标的结果文件
    result_file = os.path.join(data_dir_pred, "result.csv")
    if not os.path.isfile(result_file):
        evaluate_BraTS_folder(data_dir_pred, data_dir_gt, num_processes=6)

    # 根据评价指标的结果文件绘制图像
    # plot_evaluate(result_file, out_dir)
