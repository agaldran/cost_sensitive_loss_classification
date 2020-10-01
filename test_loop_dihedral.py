import os
import os.path as osp


experiment_list = ['ce_reg_1e2_resnext50_exp2_Mtask_corr',
                   'gls_reg_1e2_resnext50_exp2_Mtask_corr',
                   'focal_loss_reg_1e2_resnext50_exp2_Mtask_corr']

csv_out_list = ['results/' + n+'.csv' for n in experiment_list]
experiment_path_list = ['experiments/'+n for n in experiment_list]

base_cmd = 'python test_dihedral_tta.py --load_path experiments/'
for i in range(len(experiment_path_list)):
    # for t in [0, 1, 2, 3]:
    for t in [2]:
        csv_n = csv_out_list[i][:-4]+'_tta_'+str(t)+'.csv'
        cmd = base_cmd + experiment_list[i] + ' --csv_out ' + csv_n + ' --dihedral_tta ' + str(t)
        if not osp.isdir('experiments/'+experiment_list[i]):
            print(experiment_list[i] + ' does not exist')
            continue
        print(experiment_list[i], 'tta =', t)
        os.system(cmd)


print(60*'*')
experiment_list = ['ce_reg_1e2_resnext50_exp2_Mtask_corr',
                   'gls_reg_1e2_resnext50_exp2_Mtask_corr',
                   'focal_loss_reg_1e2_resnext50_exp2_Mtask_corr']
csv_out_list = ['results/' + n+'.csv' for n in experiment_list]
experiment_path_list = ['experiments/'+n for n in experiment_list]

base_cmd = 'python test_dihedral_tta.py --csv_test data/test_messidor_2.csv --load_path experiments/'
for i in range(len(experiment_path_list)):
    # for t in [0, 1, 2, 3]:
    for t in [2]:
        csv_n = csv_out_list[i][:-4]+'_tta_'+str(t)+'.csv'
        csv_n = csv_out_list[i][:-4] + '_tta_' + str(t) + '_messidor2.csv'
        cmd = base_cmd + experiment_list[i] + ' --csv_out ' + csv_n + ' --dihedral_tta ' + str(t)
        if not osp.isdir('experiments/'+experiment_list[i]):
            print(experiment_list[i] + ' does not exist')
            continue
        print(experiment_list[i], 'tta =', t)
        os.system(cmd)

# print(60*'*')
# experiment_list = ['ce_reg_1e2_resnext50_exp2_Mtask',
#                    'focal_loss_reg_1e2_resnext50_exp2_Mtask',
#                    'gls_reg_1e2_resnext50_exp2_Mtask']
# csv_out_list = ['results/' + n+'.csv' for n in experiment_list]
# experiment_path_list = ['experiments/'+n for n in experiment_list]
#
# base_cmd = 'python test_dihedral_tta.py --csv_test data/test_idrid.csv --load_path experiments/'
# for i in range(len(experiment_path_list)):
#     # for t in [0, 1, 2, 3]:
#     for t in [2]:
#         csv_n = csv_out_list[i][:-4]+'_tta_'+str(t)+'.csv'
#         csv_n = csv_out_list[i][:-4] + '_tta_' + str(t) + '_idrid.csv'
#         cmd = base_cmd + experiment_list[i] + ' --csv_out ' + csv_n + ' --dihedral_tta ' + str(t)
#         if not osp.isdir('experiments/'+experiment_list[i]):
#             print(experiment_list[i] + ' does not exist')
#             continue
#         print(experiment_list[i], 'tta =', t)
#         os.system(cmd)
#
# print(60*'*')
# experiment_list = ['ce_reg_1e2_resnext50_exp2_Mtask',
#                    'focal_loss_reg_1e2_resnext50_exp2_Mtask',
#                    'gls_reg_1e2_resnext50_exp2_Mtask']
# csv_out_list = ['results/' + n+'.csv' for n in experiment_list]
# experiment_path_list = ['experiments/'+n for n in experiment_list]
#
# base_cmd = 'python test_dihedral_tta.py --csv_test data/test_aptos.csv --load_path experiments/'
# for i in range(len(experiment_path_list)):
#     # for t in [0, 1, 2, 3]:
#     for t in [2]:
#         csv_n = csv_out_list[i][:-4]+'_tta_'+str(t)+'.csv'
#         csv_n = csv_out_list[i][:-4] + '_tta_' + str(t) + '_aptos.csv'
#         cmd = base_cmd + experiment_list[i] + ' --csv_out ' + csv_n + ' --dihedral_tta ' + str(t)
#         if not osp.isdir('experiments/'+experiment_list[i]):
#             print(experiment_list[i] + ' does not exist')
#             continue
#         print(experiment_list[i], 'tta =', t)
#         os.system(cmd)