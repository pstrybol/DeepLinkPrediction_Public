from DeepLinkPrediction.utils import *
from venn import venn2, venn
from scipy.stats import kruskal
import pandas as pd
import numpy as np
import glob
import pickle
import os
os.chdir('/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark')

diseases = ['Bile Duct Cancer', 'Brain Cancer', 'Bladder Cancer', 'Breast Cancer', 'Lung Cancer', 'Prostate Cancer',
            'Skin Cancer']
methods_nice_name_d = {'line-opene': 'LINE', 'n2v-opene': 'node2vec', 'deepwalk-opene': 'DeepWalk',
                       'grarep-opene': 'GraRep', 'original': 'DepMap', 'DLP-hadamard': 'DLP',
                       'DLP-weighted-l2-deepwalk-opene': 'DLP-DeepWalk'}
baselines = {'common-neighbours', 'jaccard-coefficient', 'adamic-adar-index', 'resource-allocation-index',
             'preferential-attachment', 'random-prediction', 'all-baselines'}

ppi_scaffold = 'STRING'
npr_ppi = 5
npr_dep = 3
pval_thresh = 0.05
drug_thresh = -2
topK = 100
train_ratio = 100
screening = ''
pos_thresh = ""

# Read in cell line information
cell_lineinfo = pd.read_csv('depmap_data/cell_line_info.csv', header=0, index_col=2)
ccle_name2depmap_id = dict(zip(cell_lineinfo.index, cell_lineinfo.DepMap_ID))
depmap_id2ccle_name = {v:k for k, v in ccle_name2depmap_id.items()}
ndex_nw_obj = read_ppi_scaffold(ppi_scaffold, "ppi_network_scaffolds/")
degree_df = ndex_nw_obj.getDegreeDF(set_index=True)

# Read in drug sensitivity data
drug_sens = pd.read_csv('depmap_data/primary-screen-replicate-collapsed-logfold-change.csv', header=0, index_col=0)
drug_sens.head()
drug_sens['ccle'] = [depmap_id2ccle_name[i] if i in depmap_id2ccle_name else np.nan for i in drug_sens.index]
drug_sens = drug_sens.loc[drug_sens.ccle.dropna().index]
drug_sens.index = drug_sens.ccle
del drug_sens['ccle']

# Read in drug sensitivity screen meta data
drug_samples = pd.read_csv('depmap_data/repurposing_samples_20200324.txt', skiprows=9, header=0, sep='\t')
broad_id_2pert_iname = dict(zip(drug_samples['broad_id'], drug_samples['pert_iname']))
drug_sens.columns = [i[0] for i in drug_sens.columns.str.split('::')]
common_drugs = set(drug_sens.columns) & set(broad_id_2pert_iname.keys())
drug_sens = drug_sens.loc[:, common_drugs]
drug_sens.columns = [broad_id_2pert_iname[i] for i in drug_sens.columns]


drug_info = pd.read_csv('depmap_data/repurposing_drugs_20200324.txt', skiprows=9, header=0, sep='\t', index_col=0)
drug_info = drug_info.loc[set([broad_id_2pert_iname[i] for i in common_drugs])]
drug_info.fillna('', inplace=True)
drug_info.to_csv('depmap_data/processed_drug_info.csv', header=True, index=True)
drug_info.loc[:, 'indication'] = drug_info.indication.str.split('|')
drug_info.loc[:, 'target'] = drug_info.target.apply(lambda x: x.split('|') if x else np.nan)
drug_info.loc[:, 'disease_area'] = drug_info.disease_area.apply(lambda x: x.split('|') if x else [])

# Construct dictionary, key = drug, value = target of corresponding drug
drug2targets_ = dict(zip(drug_info.target.dropna().index, drug_info.target.dropna().values))
drug2targets = {k:v for k, v in drug2targets_.items() if len(v) == 1}
all_targets = set.union(*[set(i) for i in drug2targets.values()])

# Saving processed DepMap files
# drug_info.to_csv('depmap_data/processed_drug_sensitivity_metadata.csv', header=True, index=True)
# drug_sens.to_csv('depmap_data/processed_primary-screen-replicate-logfold.csv', header=True, index=True)

total_sensitive_targets_retrieved = {}
percentage_sensitive_targets_retrieved = {}
fig, axs = plt.subplots(1, 1, figsize=(8.5, 3.5))
# annot = ["a", "b"]
# for i, disease in enumerate(["Lung Cancer"]):
for i, disease in enumerate(diseases):
    print(disease)
    # i, disease = 0, 'Lung Cancer'
    total_sensitive_targets_retrieved[disease] = {}
    percentage_sensitive_targets_retrieved[disease] = {}
    # for topK in np.append(np.arange(50, 550, 50), 1000):
    total_sensitive_targets_retrieved[disease][topK] = {}
    percentage_sensitive_targets_retrieved[disease][topK] = {}
    if screening == "_crispr":
        methods = sorted([f.split('/')[-1].split('_')[0] for f in
                          glob.glob(f"onlyDEPALL_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb128{screening}{pos_thresh}/"
                                    f"{disease.replace(' ', '_')}_{train_ratio}percent/*")])
    else:
        methods = sorted([f.split('/')[-1].split('_')[0] for f in
                          glob.glob(
                              f"onlyDEPALL_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb128{screening}/"
                              f"{disease.replace(' ', '_')}_{train_ratio}percent/*")])

    methods = list(set(methods) - {'CNE', 'DLP-average-deepwalk-opene',
                                   'DLP-weighted-l1-deepwalk-opene', 'DLP-hadamard-deepwalk-opene',
                                   'line-opene', 'VERSE', 'n2v-opene', 'metapath2vec++',
                                   "GraphSAGE-average", "GraphSAGE-hadamard", "GraphSAGE-weighted-l1",
                                   "GraphSAGE-weighted-l2"} - baselines)
    print(methods)
    dis_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}.csv", header=0,
                         index_col=0)
    crispr_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}_crispr{pos_thresh}.csv", header=0,
                            index_col=0)

    if screening == "_crispr":
        heterogeneous_network = pd.read_csv(f"heterogeneous_networks/"
                                            f"{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies{screening}{pos_thresh}.csv")
    else:
        heterogeneous_network = pd.read_csv(f"heterogeneous_networks/"
                                            f"{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies{screening}.csv")

    heterogeneous_network_obj = UndirectedInteractionNetwork(heterogeneous_network)

    pos = extract_pos_dict_at_threshold(dis_df, threshold=-1.5)
    all_pos = set.union(*[set(i) for i in pos.values()])

    # venn({"cancer dependencies": all_pos, "drug targets": all_targets})
    # plt.savefig(f"drug_sensitivity_data/venn_poster_deptarget_overlap", dpi=300)
    # plt.close()

    interm = extract_interm_dict_at_threshold(dis_df, pos, pos_threshold=-1.5, neg_threshold=-0.5)
    all_interm = set.union(*[set(i) for i in interm.values()]) - all_pos

    pos_neg_df = pd.DataFrame([dis_df.applymap(lambda x: x < -1.5).sum(axis=1).mean(),
                               dis_df.applymap(lambda x: (x > -1.5) & (x < -0.5)).sum(axis=1).mean(),
                               dis_df.applymap(lambda x: x > -0.5).sum(axis=1).mean()],
                              index=["Dependency", "Indecisive score", "No dependency"],
                              columns=["count"])

    # pie, ax = plt.subplots(figsize=[6.5, 4])
    # plt.pie(x=pos_neg_df.values.ravel(), autopct="%.1f%%", explode=[0.2] * 3, labels=pos_neg_df.index, pctdistance=0.5)
    # plt.title(f"Class imbalance for disease {disease}")
    # plt.savefig(f"drug_sensitivity_data/100percent_final/{disease.replace(' ', '_')}_posintermneg_pie", dpi=300)
    # plt.close()
    # plt.show()
    if screening == '_crispr':
        common_cls = set(drug_sens.index) & set(crispr_df.index)
    else:
        common_cls = set(drug_sens.index) & set(dis_df.index)

    subset_drug_sens = drug_sens.loc[common_cls, drug2targets]

    # Plot original TP/intermediaries/TN
    plot_distribution_drug_sens(drug2targets=drug2targets, all_pos=all_pos, all_interm=all_interm,
                                subset_drug_sens=subset_drug_sens, dis_df=dis_df,
                                title=disease,
                                save_raw_data=None, save_fp=None,
                                annotation=None, ax=None)
    save_fp = f"drug_sensitivity_data_{ppi_scaffold}/100percent_final/"\
              f"TP_intermediary_FP_drugsensprofiles_cellsys_{disease.replace(' ', '_')}"
    plt.savefig(save_fp, dpi=300)

    degree_dict = {}
    degree_dict["DLP-weighted-l2-deepwalk-opene"] = []
    degree_dict["preferential-attachment"] = []

    for i, cl in enumerate(common_cls):
        print(f"{i+1}/{len(common_cls)} - {cl}")
        total_sensitive_targets_retrieved[disease][topK][cl] = {}
        percentage_sensitive_targets_retrieved[disease][topK][cl] = {}

        # Calculate on per cell line basis all the drugs that are sensitive (< -2) and their corresponding targets
        cl_sensitive_drugs = subset_drug_sens[drug2targets].loc[cl][subset_drug_sens[drug2targets].loc[cl] < drug_thresh]

        if cl_sensitive_drugs.shape[0] > 0:
            if screening == "_crispr":
                total_sensitive_targets = set.union(*[set(drug2targets[k]) for k in cl_sensitive_drugs.index]) & \
                                          set(crispr_df.columns)
            else:
                total_sensitive_targets = set.union(*[set(drug2targets[k]) for k in cl_sensitive_drugs.index]) & \
                                          set(dis_df.columns)
            total_sensitive_targets = total_sensitive_targets & set(heterogeneous_network_obj.node_names)
        else:
            total_sensitive_targets = set()

        # Note save the sensitive targets per disease and per cell line
        try:
            os.makedirs(f"drug_sensitivity_data_{ppi_scaffold}/targets_per_cell_line{screening}/{disease}/")
        except:
            print("folder exists")

        with open(f"drug_sensitivity_data_{ppi_scaffold}/targets_per_cell_line{screening}/{disease}/{cl}_targets_min{abs(drug_thresh)}.txt", "w") as f:
            f.writelines([i+'\n' for i in total_sensitive_targets])

        if screening == "_crispr":
            tmp_ori_df = crispr_df
            ori_method = "CRISPR"
        else:
            tmp_ori_df = dis_df
            ori_method = "original"

        top100 = set()
        top100 = get_topK_intermediaries(tmp_ori_df, cl, top100, topK, original=True,
                                         top=True, cl_list=list(tmp_ori_df.index))

        total_sensitive_targets_retrieved[disease][topK][cl][ori_method] = top100 & total_sensitive_targets
        if len(total_sensitive_targets) > 0:
            percentage_sensitive_targets_retrieved[disease][topK][cl][ori_method] = \
                len(top100 & total_sensitive_targets) / len(total_sensitive_targets) * 100
        else:
            percentage_sensitive_targets_retrieved[disease][topK][cl][ori_method] = np.nan

        for method in methods:
        # for clr, method in zip(['b', 'g'], ["DLP-weighted-l2-deepwalk-opene", "preferential-attachment"]):
            if screening == "":
                pos_thresh = ""
        #     print(method)
            total_df = pd.read_pickle(glob.glob(f"onlyDEPALL_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb128{screening}{pos_thresh}/"
                                                f"{disease.replace(' ', '_')}_{train_ratio}percent/"
                                                f"{method}*/"
                                                f"full_df_allruns_{disease.replace(' ', '_')}_emb128_{train_ratio}percent_final.pickle")[0])

            # Get the top 100 genes based on the ranking of one of the considerd LP methods
            top100 = set()
            top100 = get_topK_intermediaries(total_df, cl, top100, topK,
                                             original=False, top=True, cl_list=list(dis_df.index))

            # degree_dict[method].append(top100)

            total_sensitive_targets_retrieved[disease][topK][cl][method] = top100 & total_sensitive_targets
            if len(total_sensitive_targets) > 0:
                percentage_sensitive_targets_retrieved[disease][topK][cl][method] = \
                    len(top100 & total_sensitive_targets)/len(total_sensitive_targets)*100
            else:
                percentage_sensitive_targets_retrieved[disease][topK][cl][method] = np.nan

            # sns.histplot(degree_df.loc[set.union(*degree_dict['DLP-weighted-l2-deepwalk-opene'])].Count, label="DLP-DeepWalk",
            #              color='b', bins=np.linspace(0, 1300), stat='density')
            # sns.histplot(degree_df.loc[set.union(*degree_dict["preferential-attachment"])].Count, label="preferential-attachment",
            #              color='g', bins=np.linspace(0, 1300), stat='density')
            # sns.histplot(np.mean(degree_dict["metapath2vec++"], axis=0).astype(int),
            #              label="metapath2vec++",
            #              color='c', bins=25)
            # plt.legend()
            # plt.title("Average degree distribution of the top 100 predictions")
            # plt.savefig(f"drug_sensitivity_data_{ppi_scaffold}/100percent_final/degree_dist_DLPvsPREFATTACH", dpi=600)
            # plt.show()

            # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
            # venn({"DLP-DeepWalk": set.union(*degree_dict['DLP-weighted-l2-deepwalk-opene']),
            #       "preferential-attachment": set.union(*degree_dict['preferential-attachment'])}, ax=axs[0],
            #      fontsize=8, legend_loc="lower center")
            # axs[0].set_title(f"Overal in top 100\nScaffold {ppi_scaffold} / screening {'rnai' if screening == '' else 'CRISPR'}",
            #                  fontsize=8)
            # venn({"DLP-DeepWalk": set.union(*degree_dict['DLP-weighted-l2-deepwalk-opene']) & all_targets,
            #       "preferential-attachment": set.union(*degree_dict['preferential-attachment']) & all_targets}, ax=axs[1],
            #      fontsize=8, legend_loc="lower center")
            # axs[1].set_title(f"Overal in targets found in top 100\nScaffold {ppi_scaffold} / screening {'rnai' if screening == '' else 'CRISPR'}",
            #                  fontsize=8)
            # # plt.show()
            # plt.savefig(f"drug_sensitivity_data_{ppi_scaffold}/100percent_final/"
            #             f"venn_poster_DLPvsprefattach_overlap_{disease.replace(' ', '_')}", dpi=300)
            # plt.close()

df_l = []
for topK, v in percentage_sensitive_targets_retrieved[disease].items():
    df_l.append(pd.DataFrame(v).mean(axis=1).to_frame(topK))

plot_df = pd.concat(df_l, axis=1)
plot_df.insert(0, 0, [0]*plot_df.shape[0])
plot_df.index = [methods_nice_name_d[k] if k in methods_nice_name_d else k for k in plot_df.index]

plot_df_melt_100 = plot_df[sorted(plot_df.columns)].iloc[:,0:11].melt(ignore_index=False)
plot_df_melt_100["method"] = plot_df_melt_100.index

plot_df_melt_1000 = plot_df[sorted(plot_df.columns)].melt(ignore_index=False)
plot_df_melt_1000["method"] = plot_df_melt_1000.index

fig, axs = plt.subplots(figsize=(6, 5))
l1 = sns.lineplot(data=plot_df_melt_1000, x="variable", y="value", hue="method", ax=axs)
l1.legend().set_title(None)
# l2 = sns.lineplot(data=plot_df_melt_1000, x="variable", y="value", hue="method", ax=axs[1])
# l2.legend().set_title(None)
axs.set_xlabel("topK threshold")
# axs[1].set_xlabel("topK threshold")
axs.set_ylabel("Percentage sensitive benchmark targets retrieved")
# axs[1].set_ylabel("Percentage sensitive benchmark targets retrieved")
plt.title(f"Thresholds on target retrieval for disease: {disease}")
# plt.show()
plt.savefig(f"drug_sensitivity_data_{ppi_scaffold}{screening}/100percent_final/"
            f"topK_thresholds_{disease.replace(' ', '_')}.pdf", dpi=600)
plt.close()

# Save the retrieved targets for all disease, all cell lines and all methods
save_fp = f"drug_sensitivity_data_{ppi_scaffold}{screening}/100percent"

try:
    os.makedirs(save_fp)
except FileExistsError:
    print("folder already exists")

# -------------------------------------- SAVE DICTS -------------------------------------- #
with open(f"{save_fp}/total_target_retrieval_all_diseases_dict.pickle", 'wb') as handle:
    pickle.dump(total_sensitive_targets_retrieved, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the retrieved targets for all disease, all cell lines and all methods
with open(f"{save_fp}/percentage_target_retrieval_all_diseases_dict.pickle", 'wb') as handle:
    pickle.dump(percentage_sensitive_targets_retrieved, handle, protocol=pickle.HIGHEST_PROTOCOL)

# -------------------------------------- LOAD DICTS -------------------------------------- #
with open(f"{save_fp}/total_target_retrieval_all_diseases_dict.pickle", 'rb') as handle:
    total_sensitive_targets_retrieved = pickle.load(handle)

# Save the retrieved targets for all disease, all cell lines and all methods
with open(f"{save_fp}/percentage_target_retrieval_all_diseases_dict.pickle", 'rb') as handle:
    percentage_sensitive_targets_retrieved = pickle.load(handle)


for disease in diseases:
    # Plot Boxplot
    total_sensitive_targets_retrieved_df = pd.DataFrame.from_dict(total_sensitive_targets_retrieved[disease]).applymap(lambda x: len(x))
    pval_ori_df = calculate_significance_vs_original(total_sensitive_targets_retrieved_df, original_method='CRISPR', fdr=0.05)
    # pval_ori_df.to_csv(f"drug_sensitivity_data_{ppi_scaffold}/100percent_final/pvals_drug_retieval_{disease.replace(' ', '_')}.csv")
    methods_significant = pval_ori_df[pval_ori_df['FDR'] < 0.05].Method.values
    print(methods_significant)

    # plot_boxplot_target_retrieval(prc_targets_d=percentage_sensitive_targets_retrieved,
    #                               disease=disease, methods_significant=methods_significant,
    #                               save_fp=f"{save_fp}/drug_target_percent_retrieval_allmethods_{disease}",
    #                               method_name_map=methods_nice_name_d,
    #                               save_raw_data=f"{save_fp}/drug_target_percent_retrieval_allmethods_{disease}_raw.csv",
    #                               pdf=True)
    plot_boxplot_target_retrieval(prc_targets_d=percentage_sensitive_targets_retrieved,
                                  disease=disease, methods_significant=methods_significant,
                                  save_fp=None,
                                  method_name_map=methods_nice_name_d,
                                  save_raw_data=None,
                                  pdf=True)


