import tarfile
import argparse

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.Task032_BraTS_2018 import convert_labels_back_to_BraTS_2018_2019_convention


def convert_all_to_BraTS(input_folder, out_folder):
    predir = os.path.dirname(input_folder)
    target_dir = join(predir, "Out_BraTS")
    print(f"output_folder : {target_dir}")
    convert_labels_back_to_BraTS_2018_2019_convention(input_folder, target_dir, num_processes=6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--pred-folder', help="模型预测结果", required=True)
    parser.add_argument("-o", '--out-folder', help="模型预测结果", required=False)
    parser.add_argument("-t", '--tar-file', help="模型预测结果", required=False)

    args = parser.parse_args()

    # 如果填写的路径是结果文件的上一级目录，批量计算
    pred_folder = args.pred_folder
    print("pred_folder: ", args.pred_folder)
    print("out_folder : ", args.out_folder)
    predict_dir_name = os.path.basename(pred_folder)
    tar_dir = f"{pred_folder}_Tar"
    print("tar_folder : ", tar_dir)

    assert predict_dir_name, "/ cant in path last"

    if "Predict" in predict_dir_name or "Ensemble" in predict_dir_name:
        folders = subdirs(pred_folder)
    else:
        folders = [pred_folder]
    
    for pred_folder in folders:
        print(pred_folder)
        data_dir_pred = join(pred_folder, "Out")
        data_dir_out = args.out_folder
        if data_dir_out is None:
            data_dir_out = join(pred_folder, "Out_BraTS")

        expected_num_cases=219
        assert len(subfiles(data_dir_pred, suffix='.nii.gz', join=False)) == expected_num_cases, "数据文件不符合要求"

        convert_all_to_BraTS(data_dir_pred, data_dir_out)

        # 将结果文件压缩为tar文件    
        pre_dir = os.path.basename(os.path.dirname(data_dir_pred))
        pre_pre_dir = os.path.basename(os.path.dirname(os.path.dirname(data_dir_pred)))
        tar_filename = f"{pre_pre_dir}_{pre_dir}.tar"
        tar_file = join(tar_dir, tar_filename)

        if not os.path.isdir(tar_dir):
            os.makedirs(tar_dir)
        if os.path.isfile(tar_file):
            os.remove(tar_file)

        cmd = f"cd {data_dir_out} && tar -cvf {tar_file} *.nii.gz"
        print(f"cmd : {cmd}")
        os.system(cmd)
        print(f"tar_file : {tar_file}")

        # with tarfile.open(tar_file, "w") as tar:
            # tar.add(data_dir_out, arcname="")
