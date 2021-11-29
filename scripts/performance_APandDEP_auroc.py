from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc
from DeepLinkPrediction.utils import read_h5py, extract_pos_dict_at_threshold
from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import re
import pickle
import argparse
import os
os.chdir('/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark')

parser = argparse.ArgumentParser()
parser.add_argument('--disease', required=True, type=str)
parser.add_argument('--screening', required=True, type=str)
parser.add_argument('--ppi_scaffold', required=True, type=str)
parser.add_argument('--pos_thresh', required=False, type=float)
parser.add_argument('--methods', required=False, type=str) # separated by ';'
args = parser.parse_args()

disease = args.disease
print(f"\n\t {disease.upper()}\n")
screening = '' if args.screening == 'rnai' else '_crispr'
ppi_scaffold = args.ppi_scaffold
pos_thresh_str = str(args.pos_thresh).replace('.', '_') if args.pos_thresh else ''
npr_ppi = 5
npr_dep = 3
train_ratio = 100
topk = 100
methods_nice_name_d = {'line-opene': 'LINE', 'n2v-opene': 'node2vec', 'deepwalk-opene': 'DeepWalk',
                       'grarep-opene': 'GraRep', 'original':'DepMap', 'DLP-hadamard': 'DLP',
                       'DLP-weighted-l2-deepwalk-opene': 'DLP-DeepWalk'}

if args.methods:
    methods = args.methods.split(';')
else:
    methods = sorted([f.split('/')[-1].split('_')[0] for f in
                      glob.glob(f"onlyDEPALL_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb128{screening}{pos_thresh_str}/"
                                f"{disease.replace(' ', '_')}_{train_ratio}percent/*")])
if 'metapath2vec++' in methods:
    methods.remove('metapath2vec++')

if disease in ["Bile Duct Cancer", "Lung Cancer", "Pan Cancer", "Skin Cancer"]:
    methods = [m for m in methods if m not in ["GraphSAGE-average", "GraphSAGE-hadamard", "GraphSAGE-weighted-l1"]]
elif disease in ["Prostate Cancer", "Bladder Cancer", "Brain Cancer"]:
    methods = [m for m in methods if m not in ["GraphSAGE-weighted-l2", "GraphSAGE-hadamard", "GraphSAGE-weighted-l1"]]
elif disease == "Breast Cancer":
    methods = [m for m in methods if m not in ["GraphSAGE-average", "GraphSAGE-weighted-l2", "GraphSAGE-weighted-l1"]]

print(methods)

all_file_loc = glob.glob(f"LP_train_test_splits{screening}/"
                         f"{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}{pos_thresh_str}/"
                         f"{disease.replace(' ', '_')}/*")

heterogeneous_network = pd.read_csv(f"/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark/"
                                    f"heterogeneous_networks/{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies{screening}{pos_thresh_str}.csv")
heterogeneous_network_obj = UndirectedInteractionNetwork(heterogeneous_network)

performance_df_ap_targets = pd.DataFrame(data=None, columns=["run0", "run1", "run2", "mean"], index=methods)
performance_df_ap_deps = pd.DataFrame(data=None, columns=["run0", "run1", "run2", "mean"], index=methods)

performance_df_auroc_targets = pd.DataFrame(data=None, columns=["run0", "run1", "run2", "mean"], index=methods)
performance_df_auroc_deps = pd.DataFrame(data=None, columns=["run0", "run1", "run2", "mean"], index=methods)

dis_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}.csv", header=0,
                     index_col=0)
crispr_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}_crispr.csv",
                            header=0, index_col=0)

# pos = extract_pos_dict_at_threshold(dis_df, threshold=-1.5)
drug_sens = pd.read_csv('depmap_data/processed_primary-screen-replicate-logfold.csv', header=0, index_col=0)
# if screening == 'crispr':
#     common_cls = set(drug_sens.index) & set(crispr_df.index)
# else:
#     common_cls = set(dis_df.index) & set(drug_sens.index)

if screening == '_crispr':
    common_cls = set(drug_sens.index) & set(crispr_df.index)
else:
    common_cls = set(drug_sens.index) & set(dis_df.index)

all_cls = set(dis_df.index) | set(crispr_df.index)
raw_data = {}

# fig_auc, ax_auc = plt.subplots(figsize=(11, 6))
for i, method in enumerate(methods):
    print(method)
    total_df = pd.read_pickle(glob.glob(f"onlyDEPALL_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb128{screening}{pos_thresh_str}/"
                                        f"{disease.replace(' ', '_')}_{train_ratio}percent/"
                                        f"{method}*/"
                                        f"full_df_allruns_{disease.replace(' ', '_')}_emb128_{train_ratio}percent_final.pickle")[0])
    total_df_subset = total_df[total_df.TestEdges_A.isin(common_cls)] # only take 88 cell lines
    total_df_subset = total_df_subset[~total_df_subset.TestEdges_B.isin(all_cls)] # remove cl2cl interaction
    total_df_subset.reset_index(drop=True, inplace=True)
    total_df_subset["labels_targets"] = np.zeros(total_df_subset.shape[0]).astype(int)

    targets = {}
    groups_cl = total_df_subset.groupby("TestEdges_A").groups
    print(len(groups_cl))
    for cl in common_cls:
        with open(f"drug_sensitivity_data_{ppi_scaffold}/targets_per_cell_line{screening}/{disease}/{cl}_targets_min2.txt", "r") as f:
            targets[cl] = set([i.strip('\n') for i in f.readlines()])
        total_df_subset.iloc[groups_cl[cl], 6] = total_df_subset.iloc[groups_cl[cl]].\
            TestEdges_B.isin(targets[cl]).astype(int)

    cl_groups = total_df_subset.groupby('TestEdges_A').groups
    to_concat = []
    for cl, ix in cl_groups.items():
        to_concat.append(total_df_subset.iloc[ix].sort_values('Mean', ascending=False))
    auc_curve_df = pd.concat(to_concat)

    if i == 0:
        raw_data["x-axis"] = np.linspace(0, 1, auc_curve_df.shape[0])

    tprs = []
    aucs = []
    for repeat in range(3):
        test_edges = read_h5py(list(filter(re.compile(rf'(.*/test_all_cls_{repeat}.hdf5)').match, all_file_loc))[0])
        test_labels = read_h5py(list(filter(re.compile(rf'(.*/label_test_all_cls_{repeat}.hdf5)').match, all_file_loc))[0])

        test_df = pd.DataFrame(test_edges, columns=['TestEdges_A', 'TestEdges_B']).\
            applymap(lambda x: heterogeneous_network_obj.int2gene[x])
        test_df["labels_deps"] = test_labels

        merged_df = pd.merge(total_df_subset, test_df, how='inner', on=['TestEdges_A', 'TestEdges_B'])
        if repeat == 0:
            performance_df_ap_targets.loc[method, "run0"] = average_precision_score(total_df_subset["labels_targets"],
                                                                                    total_df_subset["Predictions_x"])
            performance_df_auroc_targets.loc[method, "run0"] = roc_auc_score(total_df_subset["labels_targets"],
                                                                             total_df_subset["Predictions_x"])

            fpr, tpr, thresholds_auc = roc_curve(auc_curve_df["labels_targets"], auc_curve_df["Predictions_x"])
            tprs.append(np.interp(np.linspace(0, 1, auc_curve_df.shape[0]), fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(performance_df_auroc_targets.loc[method, "run0"])

            performance_df_ap_deps.loc[method, "run0"] = average_precision_score(merged_df["labels_deps"],
                                                                                 merged_df["Predictions_x"])
            performance_df_auroc_deps.loc[method, "run0"] = roc_auc_score(merged_df["labels_deps"],
                                                                          merged_df["Predictions_x"])

        elif repeat == 1:
            performance_df_ap_targets.loc[method, "run1"] = average_precision_score(total_df_subset["labels_targets"],
                                                                                    total_df_subset["Predictions_y"])
            performance_df_auroc_targets.loc[method, "run1"] = roc_auc_score(total_df_subset["labels_targets"],
                                                                             total_df_subset["Predictions_y"])

            fpr, tpr, thresholds_auc = roc_curve(auc_curve_df["labels_targets"], auc_curve_df["Predictions_y"])
            tprs.append(np.interp(np.linspace(0, 1, auc_curve_df.shape[0]), fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(performance_df_auroc_targets.loc[method, "run1"])


            performance_df_ap_deps.loc[method, "run1"] = average_precision_score(merged_df["labels_deps"],
                                                                                 merged_df["Predictions_y"])

            performance_df_auroc_deps.loc[method, "run1"] = roc_auc_score(merged_df["labels_deps"],
                                                                          merged_df["Predictions_y"])
        else:
            performance_df_ap_targets.loc[method, "run2"] = average_precision_score(total_df_subset["labels_targets"],
                                                                                    total_df_subset["Predictions"])
            performance_df_auroc_targets.loc[method, "run2"] = roc_auc_score(total_df_subset["labels_targets"],
                                                                             total_df_subset["Predictions"])

            fpr, tpr, thresholds_auc = roc_curve(auc_curve_df["labels_targets"], auc_curve_df["Predictions"])
            tprs.append(np.interp(np.linspace(0, 1, auc_curve_df.shape[0]), fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(performance_df_auroc_targets.loc[method, "run2"])

            performance_df_ap_deps.loc[method, "run2"] = average_precision_score(merged_df["labels_deps"],
                                                                                 merged_df["Predictions"])
            performance_df_auroc_deps.loc[method, "run2"] = roc_auc_score(merged_df["labels_deps"],
                                                                          merged_df["Predictions"])

        performance_df_ap_targets.loc[method, "mean"] = average_precision_score(total_df_subset["labels_targets"],
                                                                                total_df_subset["Mean"])
        performance_df_auroc_targets.loc[method, "mean"] = roc_auc_score(total_df_subset["labels_targets"],
                                                                         total_df_subset["Mean"])

        performance_df_ap_deps.loc[method, "mean"] = average_precision_score(merged_df["labels_deps"],
                                                                             merged_df["Mean"])
        performance_df_auroc_deps.loc[method, "mean"] = roc_auc_score(merged_df["labels_deps"],
                                                                      merged_df["Mean"])

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(np.linspace(0, 1, auc_curve_df.shape[0]), mean_tpr)
    mean_tpr[-1] = 1.0
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    raw_data[method] = mean_tpr

    # ax_auc.plot(np.linspace(0, 1, auc_curve_df.shape[0]), mean_tpr,
    #             label=rf'{methods_nice_name_d[method] if method in methods_nice_name_d else method} (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

# ax_auc.legend()
# ax_auc.set_title(f"{disease} - AUROC across 3 runs")
# ax_auc.set_xlabel("FPR - FP/(FP+TN)")
# ax_auc.set_ylabel("TPR - TP/(TP+FN)")
# fig_auc.tight_layout()
# plt.legend(bbox_to_anchor=(0, -0.38, 1, 0), loc="lower left",
#            mode="expand", borderaxespad=0, ncol=2)
# plt.show()
# fig_auc.savefig(f"drug_sensitivity_data/100percent_final/roc_targets", bbox_inches='tight', dpi=300)
# plt.close(fig_auc)
pd.DataFrame(raw_data).to_csv(f"drug_sensitivity_data_{ppi_scaffold}/100percent_final/roc_targets_raw_{disease.replace(' ', '_')}{screening}{pos_thresh_str}.csv")

tmp_d = {"AP_targets": performance_df_ap_targets, "AUROC_targets": performance_df_auroc_targets}

save_fp = f"CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ','_')}{pos_thresh_str}/"
try:
    os.makedirs(save_fp)
except:
    print("directory exists")

with open(save_fp+
          f"target_performance_100percent_final_{disease.replace(' ','_')}{pos_thresh_str}.pickle", 'wb') as handle:
    pickle.dump(tmp_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
