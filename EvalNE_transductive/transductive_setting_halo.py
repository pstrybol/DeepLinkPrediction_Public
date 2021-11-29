from DeepLinkPrediction.utils import *
from DeepLinkPrediction import main_DLP
import pandas as pd
import numpy as np
import pickle
import random
import glob
import re
import os
disease = 'Lung Cancer'
ppi_scaffold = "STRING"
screening = ''
pos_thresh = ''
npr_ppi = 5
npr_dep = 3
assert os.getcwd().split('/')[-1] == "EvalNE_transductive", "Wrong working directory"
BASE_PATH = '/'.join(os.getcwd().split('/')[:-1])

original_dis_df = pd.read_csv(
        f"{BASE_PATH}/depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}{screening}"
        f"{pos_thresh}.csv",
        header=0, index_col=0)
original_pos = extract_pos_dict_at_threshold(original_dis_df, threshold=-1.5)

ppi_obj = read_ppi_scaffold(ppi_scaffold, f"{BASE_PATH}/ppi_network_scaffolds/")

transductive_df = pd.read_pickle(f"{BASE_PATH}/transductive_setting/{disease.replace(' ', '_')}/"
                                f"transductive_binned_df.pickle")

with open(f"{BASE_PATH}/transductive_setting/{disease.replace(' ', '_')}/"
          f"bin2gene_halo.pickle", 'rb') as handle:
    bin2gene = pickle.load(handle)

print([len(i) for i in bin2gene.values()])

print("HALO")

bin2gene = {k: list(v) for k, v in bin2gene.items()}
total = np.sum([len(v) for v in bin2gene.values()])

genes_done = {k: set() for k in bin2gene.keys()}
i = 0
while i != total:
    for bin_ in list(bin2gene.keys()):
        i+=1
        gene = random.sample(set(bin2gene[bin_])-genes_done[bin_], 1)[0]
        print(bin_, gene, i)
        genes_done[bin_].add(gene)
        if len(genes_done[bin_]) == len(bin2gene[bin_]):
            del bin2gene[bin_]
            print(f"{bin_} key deleted")

        all_file_loc = glob.glob(f"{BASE_PATH}/LP_train_test_splits{screening}/"
                                 f"{bin_}_{gene}_{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}{pos_thresh}/"
                                 f"{disease.replace(' ', '_')}/*")
        heterogeneous_network = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                                            f"{bin_}_{gene}_{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies{screening}.csv")
        nw_obj = UndirectedInteractionNetwork(heterogeneous_network)
        cls = set(nw_obj.node_names) - set(ppi_obj.node_names)

        test_df = pd.DataFrame([[nw_obj.gene2int[gene], nw_obj.gene2int[b]] for b in cls], columns=['GeneA', 'GeneB'])
        test_df['label'] = [1 if gene in original_pos[b] else 0 for b in cls]
        # If we want to calculate performance
        test_edges = test_df[["GeneA", "GeneB"]].values
        test_labels = test_df['label'].values

        for repeat in range(3):
            print(repeat)
            train_edges = read_h5py(list(filter(re.compile(rf'(.*/trE_{repeat}.hdf5)').match, all_file_loc))[0])
            train_labels = read_h5py(list(filter(re.compile(rf'(.*/label_trE_{repeat}.hdf5)').match, all_file_loc))[0])

            train_edges_val = read_h5py(list(filter(re.compile(rf'(.*/trE_val_{repeat}.hdf5)').match, all_file_loc))[0])
            train_labels_val = read_h5py(list(filter(re.compile(rf'(.*/label_trE_val_{repeat}.hdf5)').match, all_file_loc))[0])
            test_edges_val = read_h5py(list(filter(re.compile(rf'(.*/teE_val_{repeat}.hdf5)').match, all_file_loc))[0])
            test_labels_val = read_h5py(list(filter(re.compile(rf'(.*/label_teE_val_{repeat}.hdf5)').match, all_file_loc))[0])

            evaluate_dw(ppi_scaffold=ppi_scaffold, disease=disease, screening="", train_edges=train_edges,
                        train_labels=train_labels, test_edges=test_edges,
                        test_labels=test_labels, train_edges_val=train_edges_val,
                        train_labels_val=train_labels_val, test_edges_val=test_edges_val,
                        test_labels_val=test_labels_val, repeat=repeat, npr_ppi=5, npr_dep=3,
                        pos_thresh="", save_embs=True, save_preds=None)

            dl_obj = main_DLP.main(tr_e=train_edges, tr_e_labels=train_labels, inputgraph=nw_obj,
                                   merge_method='hadamard', predifined_embs=f"deepwalk-opene_{disease.replace(' ', '_')}_{ppi_scaffold}_"
                                                                            f"embsize128_80percent{screening}{pos_thresh}" \
                                                                            f"_nprPPI{npr_ppi}_nprDEP{npr_dep}/emb_deepwalk-opene_{repeat}.tmp")

            test_df[f'predictions_rep{repeat}'] = dl_obj.predict_proba(test_edges)
        pd.to_pickle(test_df, f"{BASE_PATH}/transductive_setting/{bin_}_{gene}_predictions.pickle")


# -------------------------------------------------------- READ IN RESULTS -------------------------------------------

# predicted_df_top = pd.read_pickle(f"{BASE_PATH}/transductive_setting/{disease.replace(' ', '_')}/top5perc_results.pickle")
# predicted_df_bottom = pd.read_pickle(f"{BASE_PATH}/transductive_setting/{disease.replace(' ', '_')}/bottom5perc_results.pickle")
#
# gene2counts_total = pd.read_pickle(f"{BASE_PATH}/transductive_setting/{disease.replace(' ', '_')}/top5perc_counts.pickle")
# gene2counts_bottom = pd.read_pickle(f"{BASE_PATH}/transductive_setting/{disease.replace(' ', '_')}/bottom5perc_counts.pickle")
# gene2counts_total.update(gene2counts_bottom)
#
# performance_top_df = transductive_performance(predicted_df_top)
# performance_bottom_df = transductive_performance(predicted_df_bottom)

# TODO: plot against degree in PPI, or number of first O neighbours that are positives

# mean_perf_df = pd.DataFrame(pd.concat([performance_top_df.mean()*100, performance_bottom_df.mean()*100]),
#                             columns=['mean'])
# mean_perf_df['count'] = [gene2counts_total[i] for i in mean_perf_df.index]
# mean_perf_df['top/bottom'] = ['top' if i in predicted_df_top else 'bottom' for i in mean_perf_df.index]
# mean_perf_df['gene'] = mean_perf_df.index
#
# _, ax = plt.subplots(figsize=(8, 5))
# s = sns.scatterplot(x='count', y='mean', hue='top/bottom', data=mean_perf_df, palette='colorblind', ax=ax)
# corr, pval = spearmanr(mean_perf_df[mean_perf_df['top/bottom'] == 'top']['count'],
#                        mean_perf_df[mean_perf_df['top/bottom'] == 'top']['mean'])
# plt.title(f"{disease} - Spearman Correlation: pvalue={pval:.2e} / Spearman RC={corr:.2}")
# plt.savefig(f"{BASE_PATH}/transductive_setting/{disease.replace(' ', '_')}/inductive_5perc_results", dpi=600)
# plt.close()
# plt.show()

# melted_perf = performance_df[performance_df.mean().sort_values(ascending=False).index].melt(ignore_index=False)
# melted_perf["repeat"] = melted_perf.index
# melted_perf["value"] = melted_perf["value"]*100
#
# _, ax = plt.subplots(figsize=(8, 5))
# b = sns.barplot(x="variable", y="value", hue="repeat", data=melted_perf, ax=ax, palette="colorblind")
# b.set_xticklabels(performance_df.mean().sort_values(ascending=False).index, rotation=30, ha='right')
# b.set_ylabel("Average Precision")
# b.set_xlabel("Top 5% positive cancer dependencies")
# plt.show()
#
