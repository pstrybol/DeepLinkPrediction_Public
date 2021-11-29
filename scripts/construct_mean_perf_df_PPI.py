from DeepLinkPrediction.utils import get_mean_performance_df, plot_heatmap_performance_values
from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np
import glob
import os
os.chdir('/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark')

diseases = ['Bile Duct Cancer', 'Prostate Cancer', 'Bladder Cancer', 'Skin Cancer', 'Brain Cancer', 'Breast Cancer',
            'Lung Cancer']
metrics = ['tn', 'fp', 'fn', 'tp', 'auroc', 'precision', 'recall', 'fallout', 'miss', 'accuracy', 'f_score',
           'average_precision', 'eval_time']
methods_nice_name_d = {'line-opene': 'LINE', 'n2v-opene': 'node2vec', 'deepwalk-opene': 'DeepWalk',
                       'grarep-opene': 'GraRep', 'DLP-hadamard': 'DLP',
                       'DLP-weighted-l2-deepwalk-opene': 'DLP-DeepWalk'}
ppi_scaffold = "STRING"
npr_ppi = 5
npr_dep = 3
method = "DLP-hadamard"
train_ratio = 80
screening = ''

ap = {}
for disease in diseases:
    print(disease)
    tmp = pd.read_pickle(f"PPI_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/"
                         f"5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{disease.replace(' ', '_')}"
                         f"_complete_metricPerofrmance_emb128.pickle")
    mean_df = get_mean_performance_df(tmp, metrics)
    mean_df.drop('metapath2vec++', inplace=True)
    mean_df.index = [method.replace('_', ' ') if method not in methods_nice_name_d else methods_nice_name_d[method]
                     for method in mean_df.index]
    ap[disease] = mean_df['Average_precision']

plot_heatmap_performance_values(ap, save_fp=f"PPI_Benchmark_Res{screening}/{ppi_scaffold}/PPI_heatmap_performance",
                                save_raw_data=None, include_mean=True, pdf=False)




