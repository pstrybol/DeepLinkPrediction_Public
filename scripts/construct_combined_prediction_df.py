from functools import reduce
from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import argparse
import os
os.chdir('/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark')

parser = argparse.ArgumentParser()
parser.add_argument('--disease', required=True, type=str)
parser.add_argument('--screening', required=True, type=str)
parser.add_argument('--ppi_scaffold', required=True, type=str)
parser.add_argument('--pos_thresh', required=False, type=float)
parser.add_argument('--emb_dim', required=True, type=int, default=128)
parser.add_argument('--npr_ppi', required=True, type=int, default=5)
parser.add_argument('--npr_dep', required=True, type=int, default=3)
parser.add_argument('--train_ratio', required=True, type=int, default=100)
parser.add_argument('--pan_cancer_performance', required=False, action='store_true')
parser.add_argument('--pan_cancer_targets', required=False, action='store_true')
args = parser.parse_args()

disease = args.disease.replace(' ', '_')
print(disease+"\n")
ppi_scaffold = args.ppi_scaffold
npr_ppi = args.npr_ppi
npr_dep = args.npr_dep
train_ratio = args.train_ratio
emb_dim = args.emb_dim
screening = '' if args.screening == 'rnai' else '_crispr'
pos_thresh_str = str(args.pos_thresh).replace('.', '_') if args.pos_thresh else ''

# Load DepMap General Data
cell_lineinfo = pd.read_csv('depmap_data/cell_line_info.csv', header=0, index_col=2)
ccle_name2depmap_id = dict(zip(cell_lineinfo.index, cell_lineinfo.DepMap_ID))
depmap_id2ccle_name = {v:k for k, v in ccle_name2depmap_id.items()}

heterogeneous_network = pd.read_csv(f'heterogeneous_networks/'
                                    f'{ppi_scaffold}_{disease}_dependencies{screening}{pos_thresh_str}.csv')
heterogeneous_network_obj = UndirectedInteractionNetwork(heterogeneous_network)

# if disease == "Pan_Cancer":
#     test_preds_fp = glob.glob(f"PANCANCER_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb{emb_dim}{screening}/"
#                               f"{disease}_{train_ratio}percent/*")
# else:

if disease == 'Pan_Cancer':
    if args.pan_cancer_performance:
        test_preds_fp = glob.glob(f"EvalNE_pancancer/method_predictions_{ppi_scaffold}{screening}/*")
    else:
        test_preds_fp = glob.glob(f"EvalNE_pancancer_target_prediction/{ppi_scaffold}{screening}/*")
else:
    test_preds_fp = glob.glob(
        f"onlyDEPALL_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}"
        f"_emb{emb_dim}{screening}/"
        f"{disease}_{train_ratio}percent/*")

# test_preds_fp = glob.glob(f"PPIandDEP_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}/"
#                           f"{disease.replace(' ','_')}_emb{emb_dim}/*")

methods = set([f.split('/')[-1].split('_')[0] for f in test_preds_fp])
if 'PanCancer' in methods:
    methods.remove('PanCancer')
methods = [m for m in methods if 'GraphSAGE' in m]
print(methods)

all_df = {}
clrs = ['b', 'g', 'r']
for method in methods:
    print(method+'\n')
    all_df[method] = []

    method_fp = [f for f in test_preds_fp if method == f.split('/')[-1].split('_')[0]][0]
    if "GraphSAGE" in method:
        f_l = sorted(glob.glob(f"{method_fp}/{method.split('-')[0]}*_test_genes*"))
        f_p = sorted(glob.glob(f"{method_fp}/{method.split('-')[0]}*_test_preds*"))
    else:
        f_l = sorted(glob.glob(f'{method_fp}/{method}*_test_genes*'))
        f_p = sorted(glob.glob(f'{method_fp}/{method}*_test_preds*'))

    for repeat in range(len(f_l)):
        print(repeat)
        test_edges = np.loadtxt(f_l[repeat], delimiter=',', dtype=int)
        test_preds = np.loadtxt(f_p[repeat], delimiter=',', dtype=float)

        tot_df = pd.DataFrame({'TestEdges_A': test_edges[:, 0], 'TestEdges_B': test_edges[:, 1]})
        # print(tot_df.shape)
        tot_df = tot_df.applymap(lambda x: heterogeneous_network_obj.int2gene[int(x)])
        tot_df['Predictions'] = test_preds
        # sns.histplot(data=tot_df, x='Predictions', bins=100, label=f"Repeat {repeat}", color=clrs[repeat])
        all_df[method].append(tot_df)
    # plt.legend()
    # plt.title(f"Dependency Probability Distribution - Lung Cancer - {method}")
    # plt.savefig("drug_sensitivity_data/100percent_final/probability_dist_lung_cancer_revision")
    # plt.close()
    # plt.show()

    all_df_v2 = reduce(lambda left, right: pd.merge(left, right,
                                                    on=['TestEdges_A', 'TestEdges_B']), all_df[method])

    all_df_v2['Mean'] = all_df_v2[[i for i in list(all_df_v2) if i.startswith('Predictions')]].mean(axis=1)
    all_df_v2.to_pickle('/'.join(f_l[0].split('/')[:-1])+
                        f"/full_df_allruns_{disease}_emb{emb_dim}_{train_ratio}percent.pickle")

