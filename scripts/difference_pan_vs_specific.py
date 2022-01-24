from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
from DeepLinkPrediction.utils import read_ppi_scaffold, construct_combined_df_DLPdeepwalk, \
    extract_pos_dict_at_threshold,get_topK_intermediaries
from sklearn.metrics import average_precision_score, roc_auc_score
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
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
diseases = ['Bile Duct Cancer', 'Prostate Cancer', 'Bladder Cancer', 'Skin Cancer', 'Brain Cancer', 'Breast Cancer',
            'Lung Cancer']

pan_nw = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                     f"{ppi_scaffold}_Pan_Cancer_dependencies{screening}.csv")
pan_nw_obj = UndirectedInteractionNetwork(pan_nw)

ppi_obj = read_ppi_scaffold(ppi_scaffold, f"{BASE_PATH}/ppi_network_scaffolds/")

total_df_dlp_deepwalk_pan = construct_combined_df_DLPdeepwalk(BASE_PATH, repeats=3, ppi_scaffold=ppi_scaffold,
                                                              screening=screening, pan_nw_obj=pan_nw_obj,
                                                              ppi_obj=ppi_obj)
total_df_dlp_deepwalk_pan.drop([i for i in total_df_dlp_deepwalk_pan.columns if i.startswith('label')],
                               axis=1, inplace=True)
top100_all = {}
for disease in diseases:
    # disease = 'Lung Cancer'
    print(disease)
    if screening == '_crispr':
        crispr_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}_crispr.csv",
                                header=0, index_col=0)
    else:
        dis_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}.csv", header=0,
                             index_col=0)
    pos = extract_pos_dict_at_threshold(dis_df, threshold=-1.5)

    heterogeneous_network = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                                        f"{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies{screening}{pos_thresh}.csv")
    nw_obj = UndirectedInteractionNetwork(heterogeneous_network)
    cls = set(nw_obj.node_names) - set(ppi_obj.node_names)

    methods = set([f.split('/')[-1].split('_')[0]
                   for f in
                   glob.glob(f"{BASE_PATH}/EvalNE_pancancer_target_prediction/{ppi_scaffold}{screening}/*")])
    methods = methods - {'PanCancer', 'targetAP'}
    methods.add('DLP-DeepWalk')
    for method in ['DLP-DeepWalk']:
        # method = 'DLP-DeepWalk'
        # print(method)
        if method == 'DLP-DeepWalk':
            total_df_pan = total_df_dlp_deepwalk_pan
        else:
            total_df_pan = pd.read_pickle(glob.glob(f"{BASE_PATH}/EvalNE_pancancer_target_prediction/"
                                                    f"{ppi_scaffold}{screening}/{method}*/"
                                                    f"full_df_allruns_Pan_Cancer_emb128_"
                                                    f"{train_ratio}percent.pickle")[0])
        total_df_pan = total_df_pan[total_df_pan.TestEdges_A.isin(cls)]



        total_df = pd.read_pickle(glob.glob(
            f"onlyDEPALL_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb128{screening}{pos_thresh}/"
            f"{disease.replace(' ', '_')}_{train_ratio}percent/"
            f"DLP-weighted-l2-deepwalk-opene*/"
            f"full_df_allruns_{disease.replace(' ', '_')}_emb128_{train_ratio}percent_final.pickle")[0])
        total_df = total_df[~total_df.TestEdges_B.isin(cls)]

        assert total_df.shape == total_df_pan.shape, "Shape mismatched"

        top100_all[disease] = {}
        top100_all[disease]['Pan Cancer'] = []
        top100_all[disease]['Cancer Specific'] = []
        #union van top 100 over alle cell lines
        for cl in cls:

            # cl = 'EKVX_LUNG'
            tmp_pan = total_df_pan[total_df_pan.TestEdges_A == cl]
            tmp_pan = tmp_pan[tmp_pan.Mean >= 0.5]
            tmp_pan['label'] = [1 if i in pos[cl] else 0 for i in tmp_pan.TestEdges_B]
            top100_pan = set()
            top100_pan = get_topK_intermediaries(tmp_pan, cl, top100_pan, 100,
                                                 original=False, top=True, cl_list=cls)
            top100_all[disease]['Pan Cancer'].append(top100_pan)

            tmp = total_df[total_df.TestEdges_A == cl]
            tmp = tmp[tmp.Mean >= 0.5]
            tmp['label'] = [1 if i in pos[cl] else 0 for i in tmp.TestEdges_B]
            tmp.label.sum() / len(pos[cl])*100
            top100 = set()
            top100 = get_topK_intermediaries(tmp, cl, top100, 100, original=False,
                                             top=True, cl_list=cl)
            top100_all[disease]['Cancer Specific'].append(top100)

plot_ = pd.DataFrame(top100_all).applymap(lambda x: len(set.union(*x))).melt(ignore_index=False)
plot_['hue'] = plot_.index
# plot_['variable'] = plot_['variable'].str.p
_, ax = plt.subplots(figsize=(8, 5))
b = sns.barplot(data=plot_, x='variable', y='value', hue='hue', palette='colorblind')
b.set_xticklabels([i.replace(' Cancer', '') for i in diseases], rotation=30, ha='right')
b.spines['top'].set_visible(False)
b.spines['right'].set_visible(False)
b.spines['bottom'].set_visible(False)
b.legend_.set_title(None)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.set_ylabel("Number of unique genes in top 100\nacross all cell line", fontsize=6)
ax.set_xlabel("Cancer Type", fontsize=12)
# plt.show()
plt.savefig(f"CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/PanvsSpecific_top100_deps", dpi=600)
plt.close()



