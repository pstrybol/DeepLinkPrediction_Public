from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
from DeepLinkPrediction.utils import extract_pos_dict_at_threshold
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os

# TODO: plot against degree in PPI, or number of first O neighbours that are positives

disease = 'Lung Cancer'
ppi_scaffold = 'STRING'
screening = ''
pos_thresh = ''
BASE_PATH = "/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark"

fps = [i for i in glob.glob(f"{BASE_PATH}/transductive_setting/*") if i.split('/')[-1].startswith('bin')]

heterogeneous_network = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                                    f"{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies{screening}{pos_thresh}.csv")
nw_obj = UndirectedInteractionNetwork(heterogeneous_network)

dis_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}.csv", header=0,
                             index_col=0)

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
