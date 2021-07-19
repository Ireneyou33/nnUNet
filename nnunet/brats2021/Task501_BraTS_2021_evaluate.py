import os.path

from nnunet.dataset_conversion.Task501_BraTS_2021 import evaluate_BraTS_folder


# def plot_evaluate


if __name__ == '__main__':
    # pred = '/home/anning/Dataset/ProjData/nnunet/RESULTS_FOLDER/nnUNet/3d_fullres/Task501_BraTS2021/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/label_pred_0'
    # gt = '/home/anning/Dataset/ProjData/nnunet/RESULTS_FOLDER/nnUNet/3d_fullres/Task501_BraTS2021/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/label_val_0'

    pred = "/home/anning/Dataset/ProjData/nnunet/RESULTS_FOLDER/nnUNet/3d_fullres/Task501_BraTS2021/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/label_pred_all_0"
    gt = "/home/anning/Dataset/ProjData/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task501_BraTS2021/labelsTr"

    # 计算评价指标的结果文件
    result_file = os.path.join(pred, "result.csv")
    if not os.path.isfile(result_file):
        evaluate_BraTS_folder(pred, gt, num_processes=4)

    # 根据评价指标的结果文件绘制图像
    # plot_evaluate(result_file, out_dir)
