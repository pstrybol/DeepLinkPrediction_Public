from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gseapy as gp
import os
import pickle
os.chdir('/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark')

diseases = ['Bile Duct Cancer', 'Brain Cancer', 'Bladder Cancer', 'Breast Cancer', 'Prostate Cancer',
            'Skin Cancer', 'Lung Cancer']
crispr_thresholds = {'Bile Duct Cancer': "_pos-1_822879", 'Prostate Cancer': "_pos-2_02022", 'Bladder Cancer': "_pos-2_029239",
                     'Skin Cancer': "_pos-2_04098", 'Brain Cancer': "_pos-2_02018", 'Breast Cancer': "_pos-1_92688",
                     'Lung Cancer': "_pos-1_99309"}
npr_ppi = 5
npr_dep = 3
ppi_scaffold = "STRING"
train_ratio = 100
topk = 100
metric = 'AP'
screening = '_crispr'
# pos_thresh = "_pos-1_99309"
training_method = None

if training_method == 'PanCancer':
    ap_per_run = pd.read_csv(f"EvalNE_pancancer/method_predictions_{ppi_scaffold}{screening}/ap_per_run_PanCancer.csv",
                             header=0, index_col=0)
else:
    with open(f"CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/ap_across_diseases", 'rb') as handle:
        ap_per_run = pickle.load(handle)

baseline_methods = ['common-neighbours', 'jaccard-coefficient', 'adamic-adar-index', 'resource-allocation-index',
                    'preferential-attachment', 'random-prediction', 'all-baselines']

for disease in diseases:
    # disease = "Lung Cancer"
    print(f"\n\t {disease.upper()}\n")

    if training_method == 'PanCancer':
        with open(f"EvalNE_pancancer_target_prediction/{ppi_scaffold}{screening}/"
                  f"targetAP_alldiseases.csv", 'rb') as handle:
            tar_preformance_dict = pickle.load(handle)
    else:
        if screening == "_crispr":
            with open(f"CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ','_')}/"
                      f"target_performance_100percent_final_{disease.replace(' ','_')}{crispr_thresholds[disease]}.pickle", 'rb') as handle:
                tar_preformance_dict = pickle.load(handle)
        else:
            with open(f"CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ','_')}/"
                      f"target_performance_100percent_final_{disease.replace(' ','_')}.pickle", 'rb') as handle:
                tar_preformance_dict = pickle.load(handle)

    # heterogeneous_network = pd.read_csv(f"heterogeneous_networks/{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies{screening}{pos_thresh}.csv")
    # heterogeneous_network_obj = UndirectedInteractionNetwork(heterogeneous_network)

    dis_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}.csv", header=0,
                         index_col=0)
    crispr_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}_"
                            f"crispr{crispr_thresholds[disease]}.csv",
                            header=0, index_col=0)

    drug_sens = pd.read_csv('depmap_data/processed_primary-screen-replicate-logfold.csv', header=0, index_col=0)

    performance_dict = {}

    if training_method == 'PanCancer':
        performance_dict['AP_deps'] = pd.DataFrame(ap_per_run.loc[disease].apply(eval).to_list(),
                                                   columns=['run0', 'run1', 'run2'],
                                                   index=ap_per_run.loc[disease].index)
        performance_dict['AP_targets'] = tar_preformance_dict[disease]

    else:
        ap_per_run[disease].index = [m.replace('_', '-') for m in ap_per_run[disease].index]
        ap_per_run[disease]["mean"] = ap_per_run[disease].apply(lambda x: x.mean(), axis=1)
        ap_per_run[disease].sort_values("mean", ascending=False, inplace=True)
        ap_per_run[disease].drop("mean", axis=1, inplace=True)
        if ap_per_run[disease].index.isin(['DLP-average', 'DLP-weighted-l2', 'DLP-weighted-l1']).any():
            ap_per_run[disease].drop(['DLP-average', 'DLP-weighted-l2', 'DLP-weighted-l1'], axis=0, inplace=True)
        performance_dict['AP_deps'] = ap_per_run[disease]

        performance_dict['AP_targets'] = tar_preformance_dict['AP_targets']
    # performance_dict['AUROC_targets'] = tar_preformance_dict['AUROC_targets']

    methods_nice_name_d = {'line-opene': 'LINE', 'n2v-opene': 'node2vec', 'deepwalk-opene': 'DeepWalk',
                           'DLP-hadamard': 'DLP', 'grarep-opene': 'GraRep',
                           'original': 'RNAi', 'DLP-weighted-l2-deepwalk-opene': 'DLP-DeepWalk',
                           "GraphSAGE-average":"GraphSAGE", "GraphSAGE-weighted-l2":"GraphSAGE",
                           "GraphSAGE-weighted-l1":"GraphSAGE", "GraphSAGE-hadamard":"GraphSAGE"}
    if metric == 'AP':
        score_func = average_precision_score
    else:
        score_func = roc_auc_score

    performance_dict['AP_deps'].index = [methods_nice_name_d[i] if i in methods_nice_name_d else i
                                         for i in performance_dict['AP_deps'].index]
    performance_dict['AP_targets'].index = [methods_nice_name_d[i] if i in methods_nice_name_d else i
                                            for i in performance_dict['AP_targets'].index]

    common_index = set(performance_dict[metric + '_' + 'targets'].index) & \
                   set(performance_dict[metric + '_' + 'deps'].index)
    if 'metapath2vec++' in common_index:
        common_index.remove("metapath2vec++")
    df_dep = performance_dict[metric + '_' + 'deps'].loc[common_index]
    # ap_per_run[disease].drop("metapath2vec++", inplace=True)
    df_tar = performance_dict[metric + '_' + 'targets'].loc[common_index]
    df_tar.drop("mean", axis=1, inplace=True)

    # orig_target_scores = []
    cl_scores_rnai = []
    target_label_l_rnai = []

    cl_scores_cirspr = []
    target_label_l_cirspr = []

    targets_rnai = {}
    targets_crispr = {}
    common_cls_crispr = set(drug_sens.index) & set(crispr_df.index)
    common_cls_rnai = set(drug_sens.index) & set(dis_df.index)

    for i, cl in enumerate(common_cls_rnai):
        with open(f"drug_sensitivity_data_{ppi_scaffold}/targets_per_cell_line/{disease}/{cl}_targets_min2.txt", "r") as f1:
            targets_rnai[cl] = set([i.strip('\n') for i in f1.readlines()])
        cl_scores_rnai.append(dis_df.loc[cl].dropna().sort_values(ascending=True) * -1)
        target_label_l_rnai.append(np.array([score in targets_rnai[cl] for score in cl_scores_rnai[i].index.values]))

    for i, cl in enumerate(common_cls_crispr):
        with open(f"drug_sensitivity_data_{ppi_scaffold}/targets_per_cell_line_crispr/{disease}/{cl}_targets_min2.txt", "r") as f2:
            targets_crispr[cl] = set([i.strip('\n') for i in f2.readlines()])
        cl_scores_cirspr.append(crispr_df.loc[cl].dropna().sort_values(ascending=True)*-1)
        target_label_l_cirspr.append(np.array([score in targets_crispr[cl] for score in cl_scores_cirspr[i].index.values]))

    depmap_tar_score = score_func(np.hstack((target_label_l_rnai)), np.hstack((cl_scores_rnai)))
    crispr_tar_score = score_func(np.hstack((target_label_l_cirspr)), np.hstack((cl_scores_cirspr)))

    print(f"RNAi: {depmap_tar_score} - CRISPR: {crispr_tar_score}")
    # crispr RNAi: 0.008582442770411432 - CRISPR: 0.015080697315120666

    # finally we calculate the correlations, averaging over the three runs:
    data_dict = {'rho': [], 'pval': []}
    for run in ['run0', 'run1', 'run2']:
        rho, p_val = spearmanr(df_dep[run], df_tar[run])
        data_dict['rho'].append(rho)
        data_dict['pval'].append(p_val)

    df = pd.DataFrame(data_dict, index=['run0', 'run1', 'run2'])
    print(df.mean())

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'chartreuse', 'coral', 'darkblue',
              'darkgreen', 'gold', 'yellow']

    fig, ax = plt.subplots(figsize=(8.5, 5))
    dep_scores, tar_scores = [], []
    ax.axhline(depmap_tar_score, ls='--', color='k', label="DepMap")
    ax.axhline(crispr_tar_score, ls='-.', color='b', label="CRISPR")
    # for i, method in enumerate(ap_per_run[disease].index):
    methods = df_dep.mean(axis=1).sort_values(ascending=False).index
    for i, method in enumerate(methods):

        plot_values_tar = df_tar.loc[method].values
        plot_values_dep = df_dep.loc[method].values
        label = methods_nice_name_d[method] if method in methods_nice_name_d.keys() else method
        if label in baseline_methods:
            ax.scatter(plot_values_dep, plot_values_tar, label=label, marker='+', c=colors[i])
        else:
            ax.scatter(plot_values_dep, plot_values_tar, label=label, marker='o', c=colors[i])
        dep_scores += [plot_values_dep[2]]
        tar_scores += [plot_values_tar[2]]

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 0.035)
    ax.set_xlabel('Dependency ' + metric, fontsize=8)
    ax.set_ylabel('Target ' + metric, fontsize=8)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)

    if disease != "Lung Cancer":
        plt.title(f"{disease} - Spearman Correlation: {np.round(df.mean().rho*100)} (p-value = {np.round(df.mean().pval, 2)})")

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), prop={'size': 6})
    # plt.show()
    plt.savefig(f"drug_sensitivity_data_{ppi_scaffold}/100percent_final/"
                f"scatterplot_preformances_{disease.replace(' ', '_')}{screening}",
                bbox_inches='tight', dpi=600)
    plt.close()

    # df_tar["category"] = ["target"] * df_tar.shape[0]
    # df_dep["category"] = ["dependency"] * df_dep.shape[0]
    # pd.concat([df_tar, df_dep]).to_csv(f"drug_sensitivity_data_{ppi_scaffold}/100percent_final/"
    #                                    f"scatterplot_preformances_{disease.replace(' ', '_')}{screening}_raw.csv")
    # plt.show()


