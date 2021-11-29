from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
from sklearn.externals.joblib import Parallel, delayed
from DeepLinkPrediction.utils import *
import pandas as pd
import argparse
import random
import pickle

if os.getcwd().endswith("DepMap_DeepLinkPrediction_Benchmark"):
    BASE_PATH = '/'.join(os.getcwd().split('/'))
else:
    BASE_PATH = '/'.join(os.getcwd().split('/')[:-1])

print(BASE_PATH)

parser = argparse.ArgumentParser()
parser.add_argument('--disease', required=True, type=str)
parser.add_argument('--screening', required=True, type=str)
parser.add_argument('--ppi_scaffold', required=True, type=str)
parser.add_argument('--pos_thresh', required=True, type=float)
parser.add_argument('--transductive', required=False, action='store_true')

args = parser.parse_args()


# Set location variables
PPI_SCAFFOLD_LOC = f"{BASE_PATH}/ppi_network_scaffolds/"
SAVE_NW = f"{BASE_PATH}/heterogeneous_networks/"

# Set Disease
disease = args.disease
print(disease)

# Set Screening type
screening = '' if args.screening == 'rnai' else '_crispr'

pos_thresh = args.pos_thresh
pos_thresh_str = f"_pos{str(args.pos_thresh).replace('.', '_')}" if args.pos_thresh != -1.5 else ""
print(pos_thresh_str)
# Choose scaffold
ppi = args.ppi_scaffold

# Read in scaffold
ndex_nw_obj = read_ppi_scaffold(ppi, PPI_SCAFFOLD_LOC)

# Read in cell line info and RNAi depedency scores
cell_lineinfo = pd.read_csv(f'{BASE_PATH}/depmap_data/cell_line_info.csv', header=0, index_col=2)
depmap2ccle = dict(zip(cell_lineinfo['DepMap_ID'], cell_lineinfo.index))
dis_groups = cell_lineinfo.groupby('disease').groups

comb_rnai = read_dependencies(f'{BASE_PATH}/depmap_data/D2_combined_gene_dep_scores.csv', ndex_nw_obj)

comb_crispr = read_dependencies(f'{BASE_PATH}/depmap_data/CRISPR_gene_effect.csv', ndex_nw_obj, rnai=False)
print(comb_crispr.shape)
comb_crispr.index = [depmap2ccle[i] if i in depmap2ccle else np.nan for i in comb_crispr.index]
comb_crispr = comb_crispr[comb_crispr.index.notnull()]

# sns.distplot(comb_crispr.max(axis=1), label='max')
# sns.distplot(comb_crispr.min(axis=1), label='min')
# plt.legend()
# plt.show()

if args.screening == 'rnai':
    if disease == 'Pan Cancer':
        diseases = ['Bile Duct Cancer', 'Brain Cancer', 'Bladder Cancer', 'Breast Cancer', 'Lung Cancer', 'Prostate Cancer',
                    'Skin Cancer']
        common_cls = set(comb_rnai.index) & set.union(*[set(dis_groups[d]) for d in diseases])
    else:
        common_cls = set(dis_groups[disease]) & set(comb_rnai.index)

    dis_df = comb_rnai.loc[common_cls]
else:
    if disease == 'Pan Cancer':
        diseases = ['Bile Duct Cancer', 'Brain Cancer', 'Bladder Cancer', 'Breast Cancer', 'Lung Cancer',
                    'Prostate Cancer',
                    'Skin Cancer']
        common_cls = set(comb_crispr.index) & set.union(*[set(dis_groups[d]) for d in diseases])
    else:
        common_cls = set(dis_groups[disease]) & set(comb_crispr.index)

    dis_df = comb_crispr.loc[common_cls]

# Extract positive depedencies (i.e. score < -1.5)
pos = extract_pos_dict_at_threshold(dis_df, threshold=pos_thresh)

# bin_ = 'bin3'
# gene = 'LSM3'
if args.transductive:
    print("Transductive")
    uniks, counts = np.unique([l for sublist in pos.values() for l in sublist], return_counts=True)
    transductive_df = pd.DataFrame(uniks, columns=['gene'], index=uniks)
    transductive_df['bin_'] = pd.qcut(counts, 3, labels=['bin1', 'bin2', 'bin3'])
    transductive_df['count'] = counts
    transductive_df.to_pickle(f"{BASE_PATH}/transductive_setting/{disease.replace(' ', '_')}/transductive_binned_df.pickle")
    bin2gene = transductive_df.groupby('bin_').groups

    with open(f"{BASE_PATH}/transductive_setting/{disease.replace(' ', '_')}/"
              f"bin2gene_total.pickle", 'wb') as handle:
        pickle.dump(bin2gene, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"{BASE_PATH}/transductive_setting/{disease.replace(' ', '_')}/"
              f"bin2gene_halo.pickle", 'wb') as handle:
        pickle.dump({k: list(v)[:20] for k, v in bin2gene.items()}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    with open(f"{BASE_PATH}/transductive_setting/{disease.replace(' ', '_')}/"
              f"bin2gene_legio.pickle", 'wb') as handle:
        pickle.dump({k: list(v)[20:] for k, v in bin2gene.items()}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for bin_, genes in bin2gene.items():
        for gene in genes:
            pos_new = {k: set(v) - {gene} for k, v in pos.items()}
            to_remove = [k for k, v in pos_new.items() if len(v) < 3]
            if to_remove:
                pos_new = {k: v for k, v in pos_new.items() if k not in to_remove}
            dis_df_new = dis_df.loc[pos_new.keys(), set(dis_df.columns) - {gene}]
            pos_new_df = pd.DataFrame([tuple((k, i)) for k, v in pos_new.items() for i in v], columns=['Gene_A', 'Gene_B'])
            if gene == 'LSM3':
                print(dis_df_new.shape)
            dis_df_new.to_csv(f"{BASE_PATH}/depmap_specific_cancer_df/{bin_}_{gene}_"
                              f"{ppi}_{disease.replace(' ', '_')}{screening}{pos_thresh_str}.csv",
                              index=True, header=True)

            heterogeneous_network = pd.concat([ndex_nw_obj.getInteractionNamed(), pos_new_df])
            heterogeneous_network_obj = UndirectedInteractionNetwork(heterogeneous_network)

            heterogeneous_network_obj.getInteractionNamed().to_csv(
                f"{BASE_PATH}/heterogeneous_networks/{bin_}_{gene}_{ppi}_{disease.replace(' ', '_')}_dependencies{screening}"
                f"{pos_thresh_str}.csv",
                header=True, index=False)
            heterogeneous_network_obj.interactions.to_csv(
                f"{BASE_PATH}/heterogeneous_networks/{bin_}_{gene}_{ppi}_{disease.replace(' ', '_')}_dependenciesint{screening}"
                f"{pos_thresh_str}.csv",
                header=None, index=False)
else:
    transductive = ""
    pos_df = pd.DataFrame([tuple((k, i)) for k, v in pos.items() for i in v], columns=['Gene_A', 'Gene_B'])
    dis_df.to_csv(f"{BASE_PATH}/depmap_specific_cancer_df/"
                  f"{ppi}_{disease.replace(' ', '_')}{screening}{pos_thresh_str}.csv",
                  index=True, header=True)

    # Integrate postive depedencies with PPI scaffold and write to csv
    heterogeneous_network = pd.concat([ndex_nw_obj.getInteractionNamed(), pos_df])
    heterogeneous_network_obj = UndirectedInteractionNetwork(heterogeneous_network)
    heterogeneous_network_obj.getInteractionNamed().to_csv(
        SAVE_NW + ppi + f"_{disease.replace(' ', '_')}_dependencies{screening}"
                        f"{pos_thresh_str}.csv",
        header=True, index=False)
    heterogeneous_network_obj.interactions.to_csv(
        SAVE_NW + ppi + f"_{disease.replace(' ', '_')}_dependencies_int{screening}"
                        f"{pos_thresh_str}.csv",
        header=None, index=False)


