import numpy as np
import pandas
import pandas as pd
import pandas_profiling as pp
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # nnUnet adam
    all_scores_9713124 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9713124.csv'
    all_scores_9713349 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9713349.csv'

    all_scores_9713350 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9713350.csv'
    all_scores_9713351 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9713351.csv'

    all_scores_9713352 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9713352.csv'
    all_scores_9713353 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9713353.csv'

    all_scores_9713354 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9713354.csv'
    all_scores_9713355 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9713355.csv'

    all_scores_9713356 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9713356.csv'
    all_scores_9713357 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9713357.csv'

    all_scores_9713358 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9713358.csv'
    all_scores_9713359 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9713359.csv'

    all_scores_9713360 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9713360.csv'
    all_scores_9714037 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9714037.csv'

    all_scores_9714038 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9714038.csv'
    all_scores_9714039 = '/home/zhangyoujia/Nutstore Files/Purkinje/BraTS2021/report/RESULT/RESULT_VALIDATION/all_scores_9714039.csv'

    
    # todo_info = unet_adam_LGGH
    # s1. ProfileReport
    # info = pd.read_csv(todo_info)
    # report = pp.ProfileReport(info)
    # report.to_file(todo_info.replace('.csv', '.html'))

    # s2. model prepare
    """
    ['name', 'Dice_ET', 'Dice_TC', 'Dice_WT', 'Hausdorff95_ET', 'Hausdorff95_TC', 'Hausdorff95_WT', 'Sensitivity_ET', 'Sensitivity_TC', 'Sensitivity_WT', 
    'Specificity_ET', 'Specificity_TC', 'Specificity_WT']
    """
    model1 = pd.read_csv(all_scores_9713124)
    model2 = pd.read_csv(all_scores_9713349)

    model3 = pd.read_csv(all_scores_9713350)
    model4 = pd.read_csv(all_scores_9713351)

    model5 = pd.read_csv(all_scores_9713352)
    model6 = pd.read_csv(all_scores_9713353)

    model7 = pd.read_csv(all_scores_9713354)
    model8 = pd.read_csv(all_scores_9713355)

    model9 = pd.read_csv(all_scores_9713356)
    model10 = pd.read_csv(all_scores_9713357)

    model11 = pd.read_csv(all_scores_9713358)
    model12 = pd.read_csv(all_scores_9713359)

    model13 = pd.read_csv(all_scores_9713360)
    model14 = pd.read_csv(all_scores_9714037)

    model15 = pd.read_csv(all_scores_9714038)
    model16 = pd.read_csv(all_scores_9714039)


    model_list_str = ['all_scores_9713124', 'all_scores_9713349', 'all_scores_9713350', 'all_scores_9713351', 'all_scores_9713352', 'all_scores_9713353', 'all_scores_9713354', 
    'all_scores_9713355', 'all_scores_9713356', 'all_scores_9713357', 'all_scores_9713358', 'all_scores_9713359', 'all_scores_9713360', 'all_scores_9714037', 'all_scores_9714038',
    'all_scores_9714039']
    model_list = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13, model14, model15, model16]
    for model in model_list:
        model['Hausdorff95_WT'] = model['Hausdorff95_WT'].map(lambda x: -x+373.129)
        model['Hausdorff95_TC'] = model['Hausdorff95_TC'].map(lambda x: -x+373.129)
        model['Hausdorff95_ET'] = model['Hausdorff95_ET'].map(lambda x: -x+373.129)

    # s3 model rank
    metrics_list = ['Dice_ET', 'Dice_TC', 'Dice_WT', 'Hausdorff95_ET', 'Hausdorff95_TC', 'Hausdorff95_WT', 'Sensitivity_ET', 'Sensitivity_TC', 'Sensitivity_WT', 
    'Specificity_ET', 'Specificity_TC', 'Specificity_WT']
    data_list = []
    for index, row in model1.iterrows():
        temp = []
        temp.append(row['name'])
        for metric in metrics_list:
            score_a = row[metric]
            temp.append(score_a)
            try:
                score_b = model2[model2['name'] == row['name']][metric].values[0]
                score_c = model3[model3 ['name'] == row['name']][metric].values[0]
                score_d = model4[model4 ['name'] == row['name']][metric].values[0]
                score_e = model5[model5 ['name'] == row['name']][metric].values[0]
                score_f = model6[model6 ['name'] == row['name']][metric].values[0]
                score_g = model7[model7 ['name'] == row['name']][metric].values[0]
                score_h = model8[model8 ['name'] == row['name']][metric].values[0]                
                score_i = model9[model9 ['name'] == row['name']][metric].values[0]
                score_j = model10[model10 ['name'] == row['name']][metric].values[0]
                score_k = model11[model11 ['name'] == row['name']][metric].values[0]

                score_l = model12[model12['name'] == row['name']][metric].values[0]
                score_m = model13[model13 ['name'] == row['name']][metric].values[0]
                score_n = model14[model14 ['name'] == row['name']][metric].values[0]
                score_o = model15[model15 ['name'] == row['name']][metric].values[0]
                score_p = model16[model16 ['name'] == row['name']][metric].values[0]
                print(metric, row['name'], score_a, score_b, score_c, score_d, score_e, score_f, score_g, score_h, score_i, score_j, score_k,
                score_l, score_m, score_n, score_o, score_p)

                temp.append(score_b)
                temp.append(score_c)
                temp.append(score_d)
                temp.append(score_e)
                temp.append(score_f)
                temp.append(score_g)
                temp.append(score_h)
                temp.append(score_i)
                temp.append(score_j)
                temp.append(score_k)

                temp.append(score_l)
                temp.append(score_m)
                temp.append(score_n)
                temp.append(score_o)
                temp.append(score_p)               
                
                score_array = pd.Series([score_a, score_b, score_c, score_d, score_e, score_f, score_g, score_h, score_i, score_j, score_k,
                score_l, score_m, score_n, score_o, score_p])
                rank = score_array.rank(method='min', ascending=False).tolist()
                temp.append(rank[0])
                temp.append(rank[1])
                temp.append(rank[2])
                temp.append(rank[3])
                temp.append(rank[4])
                temp.append(rank[5])
                temp.append(rank[6])
                temp.append(rank[7])
                temp.append(rank[8])
                temp.append(rank[9])
                temp.append(rank[10])

                temp.append(rank[11])
                temp.append(rank[12])
                temp.append(rank[13])
                temp.append(rank[14])
                temp.append(rank[15])
               
                #print(temp)
                #print(temp.shape)
            except:
                print('Error!!! uid:{}'.format(row['name']))
        data_list.append(temp)

    ['name', 'Dice_ET', 'Dice_TC', 'Dice_WT', 'Hausdorff95_ET', 'Hausdorff95_TC', 'Hausdorff95_WT', 'Sensitivity_ET', 'Sensitivity_TC', 'Sensitivity_WT', 
    'Specificity_ET', 'Specificity_TC', 'Specificity_WT']

    columns1 = ['Dice_WT_{}'.format(model_i) for model_i in model_list_str]
    columns2 = ['Dice_WT_rank_{}'.format(model_i) for model_i in model_list_str]
    columns3 = ['Dice_TC_{}'.format(model_i) for model_i in model_list_str]
    columns4 = ['Dice_TC_rank_{}'.format(model_i) for model_i in model_list_str]
    columns5 = ['Dice_ET_{}'.format(model_i) for model_i in model_list_str]
    columns6 = ['Dice_ET_rank_{}'.format(model_i) for model_i in model_list_str]
    columns7 = ['Hausdorff95_WT_{}'.format(model_i) for model_i in model_list_str]
    columns8 = ['Hausdorff95_WT_rank_{}'.format(model_i) for model_i in model_list_str]
    columns9 = ['Hausdorff95_TC_{}'.format(model_i) for model_i in model_list_str]
    columns10 = ['Hausdorff95_TC_rank_{}'.format(model_i) for model_i in model_list_str]
    columns11 = ['Hausdorff95_ET_{}'.format(model_i) for model_i in model_list_str]
    columns12 = ['Hausdorff95_ET_rank_{}'.format(model_i) for model_i in model_list_str]

    columns13 = ['Sensitivity_WT_{}'.format(model_i) for model_i in model_list_str]
    columns14 = ['Sensitivity_WT_rank_{}'.format(model_i) for model_i in model_list_str]
    columns15 = ['Sensitivity_TC_{}'.format(model_i) for model_i in model_list_str]
    columns16 = ['Sensitivity_TC_rank_{}'.format(model_i) for model_i in model_list_str]
    columns17 = ['Sensitivity_ET_{}'.format(model_i) for model_i in model_list_str]
    columns18 = ['Sensitivity_ET_rank_{}'.format(model_i) for model_i in model_list_str]
    columns19 = ['Specificity_WT_{}'.format(model_i) for model_i in model_list_str]
    columns20 = ['Specificity_WT_rank_{}'.format(model_i) for model_i in model_list_str]
    columns21 = ['Specificity_TC_{}'.format(model_i) for model_i in model_list_str]
    columns22 = ['Specificity_TC_rank_{}'.format(model_i) for model_i in model_list_str]
    columns23 = ['Specificity_ET_{}'.format(model_i) for model_i in model_list_str]
    columns24 = ['Specificity_ET_rank_{}'.format(model_i) for model_i in model_list_str]

    columns = columns1 + columns2 + columns3 + columns4 + columns5 + columns6 + columns7 + columns8 + columns9 + columns10 + columns11 + columns12 + columns13 + columns14 + columns15 + columns16 + columns17 + columns18 + columns19 + columns20 + columns21 + columns22 + columns23 + columns24
    #print(columns)

    final_info = pd.DataFrame(data=data_list, columns=['sub_name'] + columns)

    # s4. calculate column mean
    mean_value = []
    rank_score = []
    rank = [0 for _ in range(len(model_list_str))]
    rank_col=[]
    rank_final=[]
    for col in final_info.columns:
        if col == 'sub_name':
            # print(col)
            mean_value.append('mean')
            rank_score.append('rank_score')
        else:
            if 'rank' in col:
                print(col, final_info[col].mean()/len(model_list_str))
                for i in range(len(model_list_str)):
                    if (model_list_str[i]==col[-len(model_list_str[i]):]):
                        rank[i]=rank[i]+final_info[col].mean()/(6*len(model_list_str))
            final_info[col] = final_info[str(col)].map(lambda x: float(x))
            mean_value.append(final_info[col].mean())
            rank_score.append(final_info[col].mean()/len(model_list_str))
    rank_array = pd.Series(rank)
    print(rank)
    print(rank_array)
    
    ranks = rank_array.rank(method='min', ascending=True).tolist()
    names=[]
    print(ranks)
    for i in range(len(final_info.columns)):
        if (i==0):
            rank_col.append('final_rank_score')
            rank_final.append('final_rank')
            names.append('names')
        if (i<len(model_list_str)+1 and i>=1):
            rank_col.append(rank[i-1])
            rank_final.append(ranks[i-1])
            names.append(model_list_str[i-1])
        if (i>=len(model_list_str)+1):
            rank_col.append(' ')
            rank_final.append(' ')
            names.append(' ')
    final_info.loc[len(final_info)] = mean_value
    final_info.loc[len(final_info)] = rank_score
    final_info.loc[len(final_info)] = names
    final_info.loc[len(final_info)] = rank_col
    final_info.loc[len(final_info)] = rank_final
    print(rank)
    # s5. save csv
    final_info.to_csv('./final_info_RESULT_VALIDATION.csv', index=False)
