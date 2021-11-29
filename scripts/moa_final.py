from venn import venn
from DeepLinkPrediction.utils import *
from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
import pandas as pd
import numpy as np
import glob
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.chdir('/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark')

ppi_scaffold = 'STRING'
disease = 'Lung Cancer'
method = 'DLP-weighted-l2-deepwalk-opene'
npr_ppi = 5
npr_dep = 3
topn = 100

dis_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}.csv",
                     header=0, index_col=0)
drug_sens = pd.read_csv('depmap_data/processed_primary-screen-replicate-logfold.csv',
                        header=0, index_col=0)
common_cls = set(dis_df.index) & set(drug_sens.index)

drug_info = pd.read_pickle("depmap_data/drug_info.pickle")
drug2targets_ = dict(zip(drug_info.target.dropna().index, drug_info.target.dropna().values))
drug2targets = {k: v for k, v in drug2targets_.items() if len(v) == 1}
drug2_targets_df = pd.DataFrame.from_dict(drug2targets, orient='index')
drug2_targets_df.columns = ["Target"]
drug2_targets_df.to_excel(f"drug_sensitivity_data/100percent_final/moa/drug2targets_df_one_target.xlsx")

all_targets_ = set.union(*[set(i) for i in drug2targets_.values()])
all_targets = set.union(*[set(i) for i in drug2targets.values()])

heterogeneous_network = pd.read_csv(f"heterogeneous_networks/"
                                    f"{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies.csv")
heterogeneous_network_obj = UndirectedInteractionNetwork(heterogeneous_network)

reactome_pathways = read_gmt_file('drug_sensitivity_data/100percent_final/moa/c2.cp.reactome.v7.2.symbols.gmt',
                                  heterogeneous_network_obj)

targets = {}
for cl in common_cls:
    with open(f"drug_sensitivity_data/targets_per_cell_line/Lung Cancer/{cl}_targets_min2.txt", "r") as f:
        targets[cl] = set([i.strip('\n') for i in f.readlines()])
all_sensitive_targets = set.union(*targets.values())

total_df_deps = pd.read_pickle(glob.glob(f"onlyDEPALL_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb128/"
                                         f"{disease.replace(' ', '_')}_100percent/{method}*/"
                                         f"full_df_allruns_{disease.replace(' ', '_')}_emb128_100percent_final.pickle")[0])
total_df_deps = total_df_deps[['TestEdges_A', 'TestEdges_B', 'Mean']]

mat = pd.read_hdf("drug_sensitivity_data/ppi2ppi_dffull.h5", key='ppi')
mat = mat.loc[dis_df.columns, dis_df.columns]

cell_gene_mat = pd.read_hdf("drug_sensitivity_data/cl2ppi_dffull.h5", key='cl')
cell_gene_mat = cell_gene_mat.loc[:, ~cell_gene_mat.columns.isin(dis_df.index)]

ppi_scaffold_object = read_ppi_scaffold('STRING', 'ppi_network_scaffolds/')
final_subnets = {}
gsea = {}
for target in ['KIF11', 'XPO1', 'VCP', 'PLK1']:
    print(target)
    new_scores = prioritization_maarten(cell_gene_mat, target, mat)
    final_subnets[target] = set(new_scores.index.values[:20])
    res = calculate_gsea(reactome_pathways, final_subnets[target], M=ppi_scaffold_object.N_nodes,
                                       pval=0.05, gene_to_include=None, prerank=False,
                                       outdir=f"drug_sensitivity_data/100percent_final/moa/{target}_GSEA_NONprerank",
                                       processes=12)
    # res.to_excel(f"drug_sensitivity_data/100percent_final/moa/reactome_gsea_{target}.xlsx")

# pd.DataFrame({"Reactome Pathways": [', '.join(gsea['KIF11']), ', '.join(gsea['XPO1']),
#                                     ', '.join(gsea['VCP']), ', '.join(gsea['PLK1'])]},
#              index=['KIF11', 'XPO1', 'VCP', 'PLK1']).to_excel("../drug_sensitivity_data/"
#                                                               "100percent_final/moa/"
#                                                               "reactome_gsea_targets_final.xlsx")
#
# for target in ['KIF11', 'XPO1', 'VCP', 'PLK1']:
#     # target = 'VCP'
#     one_subnet = heterogeneous_network_obj.subsetNetwork(nodes=final_subnets[target] | {target}).getInteractionNamed()
    # one_subnet['target_min2'] = one_subnet.Gene_A.isin(set.union(*targets.values())).values
    # one_subnet['target_03'] = one_subnet.Gene_A.isin(all_targets - set.union(*targets.values())).values
    # one_subnet.to_csv(f"drug_sensitivity_data/100percent_final/moa/{target}_subnet.csv")

# --------------------------- TOTAL SUBNET -------------------------------------------------------------------------- #
# all_genes = set.union(*final_subnets.values()) | set(final_subnets.keys())
# groups = {}
# for g in set.union(*final_subnets.values()):
#     for target, net in final_subnets.items():
#         if g in net:
#             if g in groups:
#                 groups[g].append(target)
#             else:
#                 groups[g] = [target]
#
# total_subnet = heterogeneous_network_obj.subsetNetwork(nodes=all_genes).getInteractionNamed()
# total_subnet["group"] = total_subnet.Gene_A.apply(lambda x: '_'.join(groups[x]))
# # total_subnet.to_csv(f"../drug_sensitivity_data/100percent_final/moa/total_subnet_table_louise_lung_new.csv")
#
# groups_v2 = {k: "_".join(v) for k, v in groups.items()}
# venn_dict = {}
# for k, v in groups_v2.items():
#     if v in venn_dict:
#         venn_dict[v].add(k)
#     else:
#         venn_dict[v] = {k}
# venn(final_subnets)
# # plt.savefig("drug_sensitivity_data/100percent_final/moa/moa_intersection_venn.svg", format='svg')
# plt.show()

# ------------------------------------ Table from target perspective --------------------------------------- #
target2drugs = pd.DataFrame.from_dict(drug2targets, orient='index').groupby(0).groups
focus_genes = ['TOP1', 'PLK1', 'UBE2N', 'RPL3', 'EGFR', 'PSMB1', 'VCP', 'TYMS', 'WEE1',
               'TUBB', 'HSP90AA1', 'XPO1', 'MTOR', 'KIF11', 'TOP2A', 'CCNA2', 'AURKB',
               'BIRC5', 'CHEK1', 'ATP1A1', 'AURKA'] # TODO: make this more general, show where you got this

top100_cls_depmap = {} # for each target, in which cell lines is it prioritized in the top 100 by DepMap
top100_cls_dlpdw = {} # for each target, in which cell lines is it prioritized in the top 100 by DLP-DeepWalk
not_top100_cls_depmap = {} # for each target, in which cell lines is it NOT prioritized in the top 100 by DepMap
not_top100_cls_dlpdw = {} # for each target, in which cell lines is it NOT prioritized in the top 100 by DLP-DeepWalk
for target in focus_genes:
    top100_cls_depmap[target] = []
    top100_cls_dlpdw[target] = []

    not_top100_cls_depmap[target] = []
    not_top100_cls_dlpdw[target] = []
    for cl in common_cls:
        if target in dis_df.loc[cl].sort_values(ascending=True)[:100].index:
            top100_cls_depmap[target].append(cl)
        else:
            not_top100_cls_depmap[target].append(cl)

        if target in cell_gene_mat.loc[cl].sort_values(ascending=False)[:100].index:
            top100_cls_dlpdw[target].append(cl)
        else:
            not_top100_cls_dlpdw[target].append(cl)

DCS_retrieved_sums = {}
DCS_not_retrieved_sums = {}
DCS_nonsens_retrieved_sums = {}
for target in focus_genes:
    # target = "VCP"
    print(target)
    # For a certain target, what corresponding drugs are sensitive in which cell lines
    drugsens_target = drug_sens.loc[common_cls, target2drugs[target]].applymap(lambda x: int(x < -2))
    # For a certain target, what corresponding drugs are NOT sensitive in which cell lines
    drugNONsens_target = drug_sens.loc[common_cls, target2drugs[target]].applymap(lambda x: int(x > 0.3))

    # For a certain target, in which cell lines is at least one corresponding drug sensitive
    drugsens_target["DCS"] = drugsens_target.apply(lambda x: int(np.sum(x) > 0), axis=1)
    # For a certain target, in which cell lines is at least one corresponding drug NOT sensitive
    # drugNONsens_target["DCS"] = drugNONsens_target.apply(lambda x: int(np.sum(x) > 0), axis=1)
    drugNONsens_target["DCS"] = drugNONsens_target.apply(lambda x: int(np.sum(x) == drugNONsens_target.shape[1]), axis=1)

    # For a certain target, in which cell lines is this target prioritized in the top 100 by DepMap
    drugsens_target['DepMap'] = dis_df.loc[drugsens_target.index, target].index.isin(top100_cls_depmap[target]).astype(int)
    drugNONsens_target['DepMap'] = dis_df.loc[drugNONsens_target.index, target].index.isin(top100_cls_depmap[target]).astype(int)

    # For a certain target, in which cell lines is this target prioritized in the top 100 by DLP-DeepWalk
    drugsens_target['DLP-DeepWalk'] = cell_gene_mat.loc[drugsens_target.index, target].index.isin(top100_cls_dlpdw[target]).astype(int)
    drugNONsens_target['DLP-DeepWalk'] = cell_gene_mat.loc[drugNONsens_target.index, target].index.isin(top100_cls_dlpdw[target]).astype(int)

    # For a certain target, count how many times DLP-DeepWalk (or DepMap) retrieves it in the top 100
    # of a cell line for which at least one drug targeting this target is sensitive
    dlp_dw_sum = drugsens_target[drugsens_target['DLP-DeepWalk'] == 1].DCS.sum()
    depmap_sum = drugsens_target[drugsens_target['DepMap'] == 1].DCS.sum()
    DCS_retrieved_sums[target] = [dlp_dw_sum, depmap_sum]

    # For a certain target, count how many times DLP-DeepWalk (or DepMap) does NOT retrieve it in the top 100
    # of a cell line for which at least one drug targeting this target is sensitive
    dlp_dw_sum = drugsens_target[drugsens_target['DLP-DeepWalk'] == 0].DCS.sum()
    depmap_sum = drugsens_target[drugsens_target['DepMap'] == 0].DCS.sum()
    DCS_not_retrieved_sums[target] = [dlp_dw_sum, depmap_sum]

    # For a certain target, count how many times DLP-DeepWalk (or DepMap) retrieves it in the top 100
    # of a cell line for which at least one drug targeting this target is NOT sensitive
    dlp_dw_sum = drugNONsens_target[drugNONsens_target['DLP-DeepWalk'] == 1].DCS.sum()
    depmap_sum = drugNONsens_target[drugNONsens_target['DepMap'] == 1].DCS.sum()
    DCS_nonsens_retrieved_sums[target] = [dlp_dw_sum, depmap_sum]

dcs_df = pd.DataFrame.from_dict(DCS_retrieved_sums, orient='index').fillna(0)
dcs_df.columns = ['DLP-DeepWalk', 'DepMap']
_, w_pval = wilcoxon(dcs_df['DLP-DeepWalk'], dcs_df['DepMap'], alternative="greater")
dcs_df.sort_values(['DLP-DeepWalk', 'DepMap'], ascending=False, inplace=True)
plot_df = dcs_df.melt(ignore_index=False)
plot_df.reset_index(inplace=True)
plot_df.columns = ["Target", "Method", "no. sensitive\ncell lines retrieved"]
plot_df.index = plot_df.Target + '_' + plot_df.Method

dcs_df_insensitive = pd.DataFrame.from_dict(DCS_nonsens_retrieved_sums, orient='index').fillna(0)
dcs_df_insensitive.columns = ['DLP-DeepWalk', 'DepMap']
plot_df_insensitive = dcs_df_insensitive.melt(ignore_index=False)
plot_df_insensitive.columns = ["Method", "no. non-sensitive\ncell lines retrieved"]
plot_df_insensitive["Target"] = plot_df_insensitive.index
plot_df_insensitive.index = plot_df_insensitive.index + '_' + plot_df_insensitive.Method
plot_df_insensitive = plot_df_insensitive.loc[plot_df.index]

found_by_both_sens = (plot_df["no. sensitive\ncell lines retrieved"] > 0).groupby(plot_df.Target).min() > 0
found_by_both_insens = (plot_df_insensitive["no. non-sensitive\ncell lines retrieved"] > 0)\
                           .groupby(plot_df_insensitive.Target).min() > 0


found_by_both = np.union1d(found_by_both_sens.loc[found_by_both_sens].index,
                           found_by_both_insens.loc[found_by_both_insens].index)


plot_df_both = plot_df.loc[[c in found_by_both for c in plot_df.Target]]
plot_df_single = plot_df.loc[[c not in found_by_both for c in plot_df.Target]]

plot_df_insens_both = plot_df_insensitive.loc[[c in found_by_both for c in plot_df_insensitive.Target]]
plot_df_insens_single = plot_df_insensitive.loc[[c not in found_by_both for c in plot_df_insensitive.Target]]

degree_df = ppi_scaffold_object.getDegreeDF(set_index=True)
dcs_df["Degree in PPI"] = degree_df.loc[dcs_df.index].Count
firstOnb = heterogeneous_network_obj.getNOrderNeighbors(order=1)
pos = extract_pos_dict_at_threshold(dis_df, threshold=-1.5)
all_pos = set.union(*[set(i) for i in pos.values()])
unik_pos, count_pos = np.unique([l for sublist in pos.values() for l in sublist], return_counts=True)
dcs_df["firstOnb"] = [len(set(firstOnb[i]) & all_pos) for i in dcs_df.index]
dcs_df["noPos"] = [count_pos[np.where(unik_pos == i)] for i in dcs_df.index]
dcs_df["noPos"] = dcs_df.noPos.apply(lambda x: x[0] if len(x)>0 else 0)
dcs_df["Fraction_pos_nbs"] = dcs_df["firstOnb"] / dcs_df["Degree in PPI"]

# dcs_df.to_csv("drug_sensitivity_data/100percent_final/plot_per_target_figure6_raw.csv")

corr_labels = ['AURKB', 'TOP2A', 'HSP90AA1', 'AURKA', 'CHEK1', 'UBE2N', 'ATP1A1', 'MTOR', 'TOP1', 'PSMB1', 'WEE1']

plot_df_single["Target"] = plot_df_single.Target.apply(lambda x: "CHEK1" if x == "UBE2N" else "UBE2N" if x == "CHEK1" else x)
plot_df_insens_single["Target"] = plot_df_insens_single.Target.apply(lambda x: "CHEK1" if x == "UBE2N" else "UBE2N" if x == "CHEK1" else x)

with open('drug_sensitivity_data/100percent_final/plot_fig6_raw.pickle', 'wb') as handle:
    pickle.dump({"dcs_df": dcs_df, "plot_df_single": plot_df_single, "plot_df_insens_single": plot_df_insens_single},
                handle, protocol=pickle.HIGHEST_PROTOCOL)

plot_subplot_col(plot_df_both, plot_df_single, dcs_df, axes=None, label_rot=30, add_annotation=True,
                 annotation=["a", "b"])
plt.show()
plt.savefig("drug_sensitivity_data/100percent_final/plot_per_target_figure6.pdf", bbox_inches='tight',
            dpi=300)

# Plot Supplementary Figure 6
to_plot = dcs_df[(dcs_df['DLP-DeepWalk'] == 0 ) | (dcs_df['DepMap'] == 0)]
to_plot.reset_index(drop=False, inplace=True)
to_plot["Method"] = to_plot['DLP-DeepWalk'].apply(lambda x: "DLP-DeepWalk" if x != 0 else "DepMap")
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
for ax, y in zip(axs, ["firstOnb", "noPos"]):
    print(y)
    sns.boxplot(x="Method", y=y, data=to_plot, palette="colorblind", ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.label.set_visible(False)
    if y == "firstOnb":
        ax.annotate('PSMB1', xy=(1+0.03, 44.3), xytext=(1+0.03, 44.3)) # ax.annotate('PSMB1', xy=(1+0.01, 45), xytext=(1+0.03, 40))
        ax.set_ylabel("# first order neighbhors\nthat are also\nstrong dependencies")
    if y == "noPos":
        ax.set_ylim(0, 88)
        ax.set_ylabel("# cell lines in which\nthe gene is a\n strong dependency")
# plt.show()
plt.savefig("drug_sensitivity_data/100percent_final/boxplot_firsOnb_noPos", bbox_inches='tight', dpi=300)
plt.close()
to_plot.to_csv("drug_sensitivity_data/100percent_final/boxplot_firsOnb_noPos_raw.csv")

all_targets_dis_df = all_targets & set(dis_df)
_, mwu_pval = mannwhitneyu(x=degree_df.loc[all_targets_dis_df].Count,
                           y=degree_df.loc[set(degree_df.index) - all_targets_dis_df].Count,
                           alternative="greater")


