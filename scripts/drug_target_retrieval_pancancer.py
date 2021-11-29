from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
from DeepLinkPrediction.utils import read_ppi_scaffold, construct_combined_df_DLPdeepwalk, read_h5py
from sklearn.metrics import average_precision_score, roc_auc_score
from functools import reduce
import pandas as pd
import numpy as np
import glob
import pickle
import os
import re

BASE_PATH = "/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark"

ppi_scaffold = 'STRING'
npr_ppi = 5
npr_dep = 3
pval_thresh = 0.05
drug_thresh = -2
topK = 100
train_ratio = 100
screening = ''
pos_thresh = ""
diseases = ['Bile Duct Cancer', 'Brain Cancer', 'Bladder Cancer', 'Breast Cancer', 'Lung Cancer', 'Prostate Cancer',
            'Skin Cancer']
methods_nice_name_d = {'line-opene': 'LINE', 'n2v-opene': 'node2vec', 'deepwalk-opene': 'DeepWalk',
                       'grarep-opene': 'GraRep', 'original': 'DepMap', 'DLP-hadamard': 'DLP',
                       'DLP-weighted-l2-deepwalk-opene': 'DLP-DeepWalk'}

pan_nw = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                     f"{ppi_scaffold}_Pan_Cancer_dependencies{screening}.csv")
pan_nw_obj = UndirectedInteractionNetwork(pan_nw)

ppi_obj = read_ppi_scaffold(ppi_scaffold, f"{BASE_PATH}/ppi_network_scaffolds/")

ppiINT2panINT = {v: pan_nw_obj.gene2int[k] for k, v in ppi_obj.gene2int.items()}

total_df_dlp_deepwalk = construct_combined_df_DLPdeepwalk(BASE_PATH, repeats=3, ppi_scaffold=ppi_scaffold,
                                                          screening=screening, pan_nw_obj=pan_nw_obj, ppi_obj=ppi_obj)
total_df_dlp_deepwalk.drop([i for i in total_df_dlp_deepwalk.columns if i.startswith('label')], axis=1, inplace=True)

drug_sens = pd.read_csv('depmap_data/processed_primary-screen-replicate-logfold.csv', header=0, index_col=0)

perf_diseases = {}
for disease in diseases:
    # disease = 'Lung Cancer'
    print(disease)
    if screening == '_crispr':
        crispr_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}_crispr.csv",
                                header=0, index_col=0)
        common_cls = set(drug_sens.index) & set(crispr_df.index)
    else:
        dis_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}.csv", header=0,
                             index_col=0)
        common_cls = set(drug_sens.index) & set(dis_df.index)

    heterogeneous_network = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                                        f"{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies{screening}{pos_thresh}.csv")
    nw_obj = UndirectedInteractionNetwork(heterogeneous_network)
    cls = set(nw_obj.node_names) - set(ppi_obj.node_names)

    methods = set([f.split('/')[-1].split('_')[0]
                      for f in glob.glob(f"{BASE_PATH}/EvalNE_pancancer_target_prediction/{ppi_scaffold}{screening}/*")])
    methods = methods - {'PanCancer', 'targetAP'}
    methods.add('DLP-DeepWalk')
    # methods.add('DLP-weighted-l2-deepwalk-opene')
    performance_df_ap_targets = pd.DataFrame(data=None, columns=["run0", "run1", "run2", "mean"], index=methods)
    performance_df_auroc_targets = pd.DataFrame(data=None, columns=["run0", "run1", "run2", "mean"], index=methods)

    for method in methods:
        print(method)
        if method == 'DLP-DeepWalk':
            total_df = total_df_dlp_deepwalk
        else:
            total_df = pd.read_pickle(glob.glob(f"{BASE_PATH}/EvalNE_pancancer_target_prediction/"
                                                f"{ppi_scaffold}{screening}/{method}*/"
                                                f"full_df_allruns_Pan_Cancer_emb128_"
                                                f"{train_ratio}percent.pickle")[0])
            # total_df = pd.read_pickle(glob.glob(
            #     f"onlyDEPALL_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb128{screening}{pos_thresh}/"
            #     f"{disease.replace(' ', '_')}_{train_ratio}percent/"
            #     f"{method}*/"
            #     f"full_df_allruns_{disease.replace(' ', '_')}_emb128_{train_ratio}percent_final.pickle")[0])
            total_df.columns = ['TestEdges_A', 'TestEdges_B', 'predictions_rep0', 'predictions_rep1',
                                'predictions_rep2', 'Mean']

        total_df = total_df[total_df.TestEdges_A.isin(common_cls)]
        total_df = total_df[total_df.TestEdges_B.isin(ppi_obj.node_names)]
        total_df.reset_index(drop=True, inplace=True)
        total_df["labels_targets"] = np.zeros(total_df.shape[0]).astype(int)

        targets = {}
        groups_cl = total_df.groupby("TestEdges_A").groups
        # print(len(groups_cl))
        for cl in common_cls:
            with open(f"drug_sensitivity_data_{ppi_scaffold}/targets_per_cell_line{screening}/"
                      f"{disease}/{cl}_targets_min2.txt", "r") as f:
                targets[cl] = set([i.strip('\n') for i in f.readlines()])
            total_df.iloc[groups_cl[cl], -1] = total_df.iloc[groups_cl[cl]]. \
                TestEdges_B.isin(targets[cl]).astype(int)
        for repeat in range(3):
            performance_df_ap_targets.loc[method,
                                          f"run{repeat}"] = average_precision_score(total_df["labels_targets"],
                                                                                    total_df[f"predictions_rep{repeat}"])
            performance_df_auroc_targets.loc[method,
                                             f"run{repeat}"] = roc_auc_score(total_df["labels_targets"],
                                                                             total_df[f"predictions_rep{repeat}"])
        performance_df_ap_targets['mean'] = performance_df_ap_targets[['run0', 'run1', 'run2']].mean(axis=1)
        perf_diseases[disease] = performance_df_ap_targets

with open(f"{BASE_PATH}/EvalNE_pancancer_target_prediction/{ppi_scaffold}{screening}/"
          f"targetAP_alldiseases.csv", 'wb') as handle:
    pickle.dump(perf_diseases, handle, protocol=pickle.HIGHEST_PROTOCOL)

