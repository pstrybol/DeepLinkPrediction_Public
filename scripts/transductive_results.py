from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
from DeepLinkPrediction.utils import extract_pos_dict_at_threshold
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os

disease = 'Lung Cancer'
ppi_scaffold = 'STRING'
screening = ''
pos_thresh = ''
BASE_PATH = "/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark"

# ---------------------------------------------------------------------------------------------------------------------
dis_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}{screening}{pos_thresh}.csv",
                     header=0, index_col=0)
heterogeneous_network = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                                    f"{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies{screening}{pos_thresh}.csv")
nw_obj = UndirectedInteractionNetwork(heterogeneous_network)
fps = [i for i in glob.glob(f"{BASE_PATH}/transductive_setting/*") if i.split('/')[-1].startswith('bin')]

transductive_df = pd.read_pickle(f"{BASE_PATH}/transductive_setting/{disease.replace(' ', '_')}/"
                                f"transductive_binned_df.pickle")

firstOnb = nw_obj.getNOrderNeighbors(order=1)
pos = extract_pos_dict_at_threshold(dis_df, threshold=-1.5)
all_pos = set.union(*[set(i) for i in pos.values()])
unik_pos, count_pos = np.unique([l for sublist in pos.values() for l in sublist], return_counts=True)
degree_df = nw_obj.getDegreeDF(set_index=True)

bin_performance = {}
bin_performance['bin1'] = {}
bin_performance['bin2'] = {}
bin_performance['bin3'] = {}
for fp in fps:
    bin_ = fp.split('/')[-1].split('_')[0]
    gene = fp.split('/')[-1].split('_')[1]
    print(bin_, gene)

    total_df = pd.read_pickle(fp)
    bin_performance[bin_][gene] = np.mean([average_precision_score(total_df.label,
                                                           total_df[f'predictions_rep{repeat}'])*100
                                   for repeat in range(3)])

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
sns.scatterplot(x=[len(set(firstOnb[i]) & all_pos) for i in bin_performance['bin1']], y=bin_performance['bin1'].values(),
                label='bin1', ax=axes[0])
sns.scatterplot(x=[len(set(firstOnb[i]) & all_pos) for i in bin_performance['bin2']], y=bin_performance['bin2'].values(),
                label='bin2', ax=axes[0])
sns.scatterplot(x=[len(set(firstOnb[i]) & all_pos) for i in bin_performance['bin3']], y=bin_performance['bin3'].values(),
                label='bin3', ax=axes[0])
axes[0].set_title("FirstOnb's that are positive\nvs Perofrmance (AP)")
sns.scatterplot(x=[transductive_df.loc[i, 'count'] for i in bin_performance['bin1']], y=bin_performance['bin1'].values(),
                label='bin1', ax=axes[1])
sns.scatterplot(x=[transductive_df.loc[i, 'count'] for i in bin_performance['bin2']], y=bin_performance['bin2'].values(),
                label='bin2', ax=axes[1])
sns.scatterplot(x=[transductive_df.loc[i, 'count'] for i in bin_performance['bin3']], y=bin_performance['bin3'].values(),
                label='bin3', ax=axes[1])
axes[1].set_title("How many times a gene is positive\nvs Perofrmance (AP)")

sns.scatterplot(x=[degree_df.loc[i, 'Count'] for i in bin_performance['bin1']], y=bin_performance['bin1'].values(),
                label='bin1', ax=axes[2])
sns.scatterplot(x=[degree_df.loc[i, 'Count'] for i in bin_performance['bin2']], y=bin_performance['bin2'].values(),
                label='bin2', ax=axes[2])
sns.scatterplot(x=[degree_df.loc[i, 'Count'] for i in bin_performance['bin3']], y=bin_performance['bin3'].values(),
                label='bin3', ax=axes[2])
axes[2].set_title("Degree of left out gene\nvs Perofrmance (AP)")
plt.legend()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------

heterogeneous_network = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                                    f"transductive20_{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies{screening}.csv")
nw_obj = UndirectedInteractionNetwork(heterogeneous_network)

ap_per_cl = {}
ap_per_gene = {}
labels_cl = {}
labels_gene = {}
for method in ['DLP-hadamard', 'DLP-DeepWalk']:
    ap_per_cl[method] = {}
    ap_per_gene[method] = {}
    total_df = pd.read_pickle(f"{BASE_PATH}/transductive_setting_20perc/{ppi_scaffold}{screening}/{disease}/"
                              f"{method}_predictions.pickle")
    total_df.reset_index(drop=True, inplace=True)
    total_df[['GeneA', 'GeneB']] = total_df[['GeneA', 'GeneB']].applymap(lambda x: nw_obj.int2gene[x])
    total_df['mean'] = total_df[['predictions_rep0', 'predictions_rep1', 'predictions_rep2']].mean(axis=1)
    cl_groups = total_df.groupby('GeneB').groups
    gene_groups = total_df.groupby('GeneA').groups

    for cl, ix in cl_groups.items():
        if total_df.loc[ix, 'label'].sum() > 0:
            labels_cl[cl] = total_df.loc[ix].sort_values('GeneA')['label'].values
            ap_per_cl[method][cl] = average_precision_score(total_df.loc[ix, 'label'], total_df.loc[ix, 'mean']) * 100

    for gene, ix in gene_groups.items():
        assert total_df.loc[ix].sort_values('GeneB')['label'].sum() > 0,'error'
        labels_gene[gene] = total_df.loc[ix].sort_values('GeneB')['label'].values
        ap_per_gene[method][gene] = average_precision_score(total_df.loc[ix, 'label'], total_df.loc[ix, 'mean']) * 100

for method in ['AROPE', 'adamic-adar-index', 'all-baselines', 'common-neighbours', 'deepwalk-opene',
               'jaccard-coefficient', 'preferential-attachment', 'random-prediction', 'resource-allocation-index']:
    ap_per_cl[method] = {}
    ap_per_gene[method] = {}
    total_df1 = pd.read_pickle(glob.glob(f"{BASE_PATH}/transductive_setting_20perc/{ppi_scaffold}{screening}/{disease}/"
                                         f"{method}*/"
                                         f"full_df_allruns_{disease}_emb128_80percent.pickle")[0])
    cl_groups1 = total_df1.groupby('TestEdges_B').groups
    gene_groups1 = total_df1.groupby('TestEdges_A').groups

    for cl, ix in cl_groups1.items():
        if cl in ap_per_cl['DLP-hadamard']:
            tmp = total_df1.loc[ix].sort_values('TestEdges_A')
            ap_per_cl[method][cl] = average_precision_score(labels_cl[cl], tmp['Mean']) * 100

    for gene, ix in gene_groups1.items():
        tmp = total_df1.loc[ix].sort_values('TestEdges_B')
        ap_per_gene[method][gene] = average_precision_score(labels_gene[gene], tmp['Mean']) * 100

# Per cell line performance
to_plot = pd.DataFrame(ap_per_cl)
to_plot = to_plot[to_plot.mean().sort_values(ascending=False).index]
sns.boxplot(data=to_plot.melt(), x='value', y='variable')
plt.show()

# Per gene performance
to_plot = pd.DataFrame(ap_per_gene)
to_plot = to_plot[to_plot.mean().sort_values(ascending=False).index]
sns.boxplot(data=to_plot.melt(), x='value', y='variable')
plt.show()


