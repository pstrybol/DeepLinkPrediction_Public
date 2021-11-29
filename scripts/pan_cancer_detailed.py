import pandas as pd
import glob

disease = "Pan Cancer"
methods_nice_name_d = {'line-opene': 'LINE', 'n2v-opene': 'node2vec', 'deepwalk-opene': 'DeepWalk',
                       'grarep-opene': 'GraRep', 'original': 'DepMap', 'DLP-hadamard': 'DLP',
                       'DLP-weighted-l2-deepwalk-opene': 'DLP-DeepWalk'}
baselines = {'common-neighbours', 'jaccard-coefficient', 'adamic-adar-index', 'resource-allocation-index',
             'preferential-attachment', 'random-prediction', 'all-baselines'}

ppi_scaffold = 'STRING'
npr_ppi = 5
npr_dep = 3
# pval_thresh = 0.05
# drug_thresh = -2
# topK = 100
train_ratio = 80

methods = sorted([f.split('/')[-1].split('_')[0] for f in
                      glob.glob(f"PANCANCER_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb128/"
                                f"{disease.replace(' ', '_')}_{train_ratio}percent/*")])

method = "DLP-hadamard"

total_df = pd.read_pickle(glob.glob(f"PANCANCER_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb128/"
                                    f"{disease.replace(' ', '_')}_{train_ratio}percent/"
                                    f"{method}*/"
                                    f"full_df_allruns_{disease.replace(' ', '_')}_emb128_{train_ratio}percent.pickle")[0])
