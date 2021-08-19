import numpy as np
import pandas
import pandas as pd
import pandas_profiling as pp
import seaborn as sns

sns.set(color_codes=True)
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # nnUnet adam
    nnunet_adam = '/home/hqy/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_2021-07-23/nnUNetTrainerV2BraTS_Adam_501/results.csv'
    # LGG
    nnunet_adam_LGG = '/home/hqy/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_2021-07-23/nnUNetTrainerV2BraTS_Adam_503_LGG/results.csv'
    # HGG
    nnunet_adam_HGG = '/home/hqy/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_2021-07-23/nnUNetTrainerV2BraTS_Adam_502_HGG/results.csv'
    # Unet++
    nnunet_adam_unetplus = '/home/hqy/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_2021-07-23/nnUNetPlusPlusTrainerV2BraTS_Adam/results.csv'
    # CoTr_IN_LeakyReLU
    CoTr_IN_LeakyReLU = '/home/hqy/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_2021-07-23/CoTr_IN_LeakyReLU/results.csv'

    DA3_BN_BD_Adam = '/home/hqy/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_2021-07-23/nnUNetTrainerV2BraTSRegions_DA3_BN_BD_Adam_501/results.csv'
    DA4_BN_Adam = '/home/hqy/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_2021-07-23/nnUNetTrainerV2BraTSRegions_DA4_BN_Adam_501/results.csv'
    DA4_BN_BD_Adam = '/home/hqy/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_2021-07-23/nnUNetTrainerV2BraTSRegions_DA4_BN_BD_Adam_501/results.csv'

    # todo_info = unet_adam_LGGHGG
    # s1. ProfileReport
    # info = pd.read_csv(todo_info)
    # report = pp.ProfileReport(info)
    # report.to_file(todo_info.replace('.csv', '.html'))

    # s2. model prepare
    """
    ['name', 'dc_whole', 'dc_core', 'dc_enh', 'hd95_whole', 'hd95_core', 'hd95_enh']
    """
    model1 = pd.read_csv(nnunet_adam)
    model2_1 = pd.read_csv(nnunet_adam_LGG)
    model2_2 = pd.read_csv(nnunet_adam_HGG)
    model2 = pd.concat([model2_1, model2_2])
    model3 = pd.read_csv(nnunet_adam_unetplus)
    model4 = pd.read_csv(CoTr_IN_LeakyReLU)
    model5 = pd.read_csv(DA3_BN_BD_Adam)
    model6 = pd.read_csv(DA4_BN_Adam)
    model7 = pd.read_csv(DA4_BN_BD_Adam)

    model_list_str = ['nnunet_adam', 'nnunet_adam_grading', 'adam_unetplus', 'CoTr_IN_LeakyReLU', 'DA3_BN_BD_Adam',
                      'DA4_BN_Adam', 'DA4_BN_BD_Adam']
    model_list = [model1, model2, model3, model4, model5, model6, model7]
    for model in model_list:
        model['hd95_whole'] = model['hd95_whole'].map(lambda x: -x + 373.129)
        model['hd95_core'] = model['hd95_core'].map(lambda x: -x + 373.129)
        model['hd95_enh'] = model['hd95_enh'].map(lambda x: -x + 373.129)

    # s3 model rank
    metrics_list = ['dc_whole', 'dc_core', 'dc_enh', 'hd95_whole', 'hd95_core', 'hd95_enh']
    data_list = []
    for index, row in model1.iterrows():
        temp = []
        temp.append(row['name'])
        for metric in metrics_list:
            score_a = row[metric]
            temp.append(score_a)
            try:
                score_b = model2[model2['name'] == row['name']][metric].values[0]
                score_c = model3[model3['name'] == row['name']][metric].values[0]
                score_d = model4[model4['name'] == row['name']][metric].values[0]
                score_e = model5[model5['name'] == row['name']][metric].values[0]
                score_f = model6[model6['name'] == row['name']][metric].values[0]
                score_g = model7[model7['name'] == row['name']][metric].values[0]
                print(metric, row['name'], score_a, score_b, score_c, score_d, score_e, score_f, score_g)
                temp.append(score_b)
                temp.append(score_c)
                temp.append(score_d)
                temp.append(score_e)
                temp.append(score_f)
                temp.append(score_g)
                score_array = pd.Series([score_a, score_b, score_c, score_d, score_e, score_f, score_g])
                rank = score_array.rank(method='min', ascending=False).tolist()
                temp.append(rank[0])
                temp.append(rank[1])
                temp.append(rank[2])
                temp.append(rank[3])
                temp.append(rank[4])
                temp.append(rank[5])
                temp.append(rank[6])
                # print(temp)
                # print(temp.shape)
            except:
                print('Error!!! uid:{}'.format(row['name']))
        data_list.append(temp)

    columns1 = ['dc_whole_{}'.format(model_i) for model_i in model_list_str]
    columns2 = ['dc_whole_rank_{}'.format(model_i) for model_i in model_list_str]
    columns3 = ['dc_core_{}'.format(model_i) for model_i in model_list_str]
    columns4 = ['dc_core_rank_{}'.format(model_i) for model_i in model_list_str]
    columns5 = ['dc_enh_{}'.format(model_i) for model_i in model_list_str]
    columns6 = ['dc_enh_rank_{}'.format(model_i) for model_i in model_list_str]
    columns7 = ['hd95_whole_{}'.format(model_i) for model_i in model_list_str]
    columns8 = ['hd95_whole_rank_{}'.format(model_i) for model_i in model_list_str]
    columns9 = ['hd95_core_{}'.format(model_i) for model_i in model_list_str]
    columns10 = ['hd95_core_rank_{}'.format(model_i) for model_i in model_list_str]
    columns11 = ['hd95_enh_{}'.format(model_i) for model_i in model_list_str]
    columns12 = ['hd95_enh_rank_{}'.format(model_i) for model_i in model_list_str]
    columns = columns1 + columns2 + columns3 + columns4 + columns5 + columns6 + columns7 + columns8 + columns9 + columns10 + columns11 + columns12
    # print(columns)

    final_info = pd.DataFrame(data=data_list, columns=['sub_name'] + columns)

    # s4. calculate column mean
    mean_value = []
    rank_score = []
    rank = [0 for _ in range(len(model_list_str))]
    rank_col = []
    rank_final = []
    for col in final_info.columns:
        if col == 'sub_name':
            # print(col)
            mean_value.append('mean')
            rank_score.append('rank_score')
        else:
            if 'rank' in col:
                print(col, final_info[col].mean() / len(model_list_str))
                for i in range(len(model_list_str)):
                    if (model_list_str[i] == col[-len(model_list_str[i]):]):
                        rank[i] = rank[i] + final_info[col].mean() / (6 * len(model_list_str))
            final_info[col] = final_info[str(col)].map(lambda x: float(x))
            mean_value.append(final_info[col].mean())
            rank_score.append(final_info[col].mean() / len(model_list_str))
    rank_array = pd.Series(rank)
    print(rank)
    print(rank_array)

    ranks = rank_array.rank(method='min', ascending=True).tolist()
    print(ranks)
    for i in range(len(final_info.columns)):
        if (i == 0):
            rank_col.append('final_rank_score')
            rank_final.append('final_rank')
        if (i < len(model_list_str) + 1 and i >= 1):
            rank_col.append(rank[i - 1])
            rank_final.append(ranks[i - 1])
        if (i >= len(model_list_str) + 1):
            rank_col.append(' ')
            rank_final.append(' ')
    final_info.loc[len(final_info)] = mean_value
    final_info.loc[len(final_info)] = rank_score
    final_info.loc[len(final_info)] = rank_col
    final_info.loc[len(final_info)] = rank_final
    print(rank)
    # s5. save csv
    final_info.to_csv('./final_info.csv', index=False)
