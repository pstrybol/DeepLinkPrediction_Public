import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

plot_df = pd.read_csv("drug_sensitivity_data_STRING/100percent_final/plot_fig6_onlyDF_raw.csv", index_col=0)
one_method_only = plot_df.loc[(plot_df['DLP-DeepWalk'] == 0) | (plot_df['DepMap'] == 0)]
other_gene_df = plot_df.loc[~plot_df.index.isin(one_method_only.index)]

fig, ax = plt.subplots()

for i in range(one_method_only.shape[0]):
    y, x = one_method_only.iloc[i, 0] - one_method_only.iloc[i, 1], one_method_only.iloc[i, 2]
    ax.scatter(x, y, c="tab:blue", ec='k', alpha=0.5)
    gene = one_method_only.index.values[i]
    if gene == 'WEE1':
        ax.annotate(gene, xy=(x + 5, y - 0.5))

    elif gene == 'TOP1':
        ax.annotate(gene, xy=(x - 14, y + 1))

    elif gene == 'TOP2A':
        ax.annotate(gene, xy=(x - 14, y + 1))

    elif gene == 'PSMB1':
        ax.annotate(gene, xy=(x - 15, y + 1))

    elif gene == 'MTOR':
        ax.annotate(gene, xy=(x + 5, y - 0.5))

    elif gene == 'HSP90AA1':
        ax.annotate(gene, xy=(x - 40, y + 1))

    elif gene == 'TYMS':
        ax.annotate(gene, xy=(x - 20, y + 1))

    else:
        ax.annotate(gene, xy=(x + 0.3, y + 1))

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.set_xlabel("First order neighbors", fontsize=13)
ax.set_ylabel("Difference in cell lines retrieved", fontsize=13)


fig, ax = plt.subplots()

for i in range(plot_df.shape[0]):
    y, x = plot_df.iloc[i, 0] - plot_df.iloc[i, 1], plot_df.iloc[i, 2]
    if y > 0:
        marker = 'o'
        c = 'tab:blue'

    elif y < 0:
        marker = 's'
        c = 'tab:orange'

    else:
        marker = 'x'
        c = 'k'

    ax.scatter(x, y, marker=marker, c=c, ec='k', alpha=0.5)
    gene = plot_df.index.values[i]
    if gene == 'WEE1':
        ax.annotate(gene, xy=(x + 5, y - 0.5))

    elif gene == 'TOP1':
        ax.annotate(gene, xy=(x - 14, y + 1))

    elif gene == 'CHEK1':
        ax.annotate(gene, xy=(x - 10, y - 3))

    elif gene == 'TOP2A':
        ax.annotate(gene, xy=(x - 14, y + 1))

    elif gene == 'PSMB1':
        ax.annotate(gene, xy=(x - 15, y + 1))

    elif gene == 'MTOR':
        ax.annotate(gene, xy=(x - 20 , y -3))

    elif gene == 'HSP90AA1':
        ax.annotate(gene, xy=(x - 40, y + 1))

    elif gene == 'TYMS':
        ax.annotate(gene, xy=(x - 16, y + 1))

    elif gene == 'AURKA':
        ax.annotate(gene, xy=(x - 30, y + 1))

    elif gene == 'EGFR':
        ax.annotate(gene, xy=(x , y - 3))

    elif gene == 'TUBB':
        ax.annotate(gene, xy=(x -15 , y - 3))

    elif gene == 'VCP':
        ax.annotate(gene, xy=(x - 8 , y + 1))

    elif gene == 'CCNA2':
        ax.annotate(gene, xy=(x + 4, y - 2))

    elif gene == 'BIRC5':
        ax.annotate(gene, xy=(x + 4, y + 1))

    elif gene == 'KIF11':
        ax.annotate(gene, xy=(x - 23, y + 1))

    elif gene == 'UBE2N':
        ax.annotate(gene, xy=(x - 20, y + 1))

    else:
        ax.annotate(gene, xy=(x + 1, y + 0.8))

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.set_xlabel("# Adjacent dependencies", fontsize=8)
ax.set_ylabel("Difference in cell lines retrieved", fontsize=8)

blue = mlines.Line2D([], [], color='tab:blue', marker='o', linestyle='None', label='DLP-DeepWalk', alpha=0.5)
orange = mlines.Line2D([], [], color='tab:orange', marker='s', linestyle='None',  label='RNAi', alpha=0.5)

ax.legend(handles=[blue, orange])
# plt.show()
# plt.savefig("drug_sensitivity_data_STRING/100percent_final/scatterplot_figure6_revised.png", dpi=600)
plt.savefig("drug_sensitivity_data_STRING/100percent_final/scatterplot_figure6_revised.pdf", dpi=600)

plot_df['difference'] = plot_df['DLP-DeepWalk'] - plot_df['DepMap']
plot_df['N_neighbors'] = plot_df['firstOnb'] / plot_df['Fraction_pos_nbs']
corr_mat = plot_df.corr("spearman")