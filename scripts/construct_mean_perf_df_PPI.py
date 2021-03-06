from DeepLinkPrediction.utils import get_mean_performance_df, plot_heatmap_performance_values, best_edgeembed_graphsage
from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np
import glob
import os
os.chdir('/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark')


metrics = ['tn', 'fp', 'fn', 'tp', 'auroc', 'precision', 'recall', 'fallout', 'miss', 'accuracy', 'f_score',
           'average_precision', 'eval_time']
methods_nice_name_d = {'line-opene': 'LINE', 'n2v-opene': 'node2vec', 'deepwalk-opene': 'DeepWalk',
                       'grarep-opene': 'GraRep', 'DLP-hadamard': 'DLP',
                       'DLP-weighted-l2-deepwalk-opene': 'DLP-DeepWalk'}
ppi_scaffold = "STRING"
npr_ppi = 5
npr_dep = 3
train_ratio = 80

ap = {}
for screening, ppi_scaffold in zip(['', '_crispr', ''], ['STRING', 'STRING', 'reactome']):
    st = 'rnai' if screening == '' else 'crispr'
    if screening == '':
        st = 'rnai'
        diseases = ['Bile Duct Cancer', 'Prostate Cancer', 'Bladder Cancer', 'Skin Cancer', 'Brain Cancer',
                    'Breast Cancer',
                    'Lung Cancer', 'Pan Cancer']
    else:
        st = 'crispr'
        diseases = ['Prostate Cancer', 'Bile Duct Cancer', 'Bladder Cancer', 'Breast Cancer', 'Skin Cancer',
                    'Brain Cancer',
                    'Lung Cancer', 'Pan Cancer']
    ap[f"{ppi_scaffold}_{st}"] = {}

    for disease in diseases:
        print(disease)
        tmp1 = pd.read_pickle(f"PPI_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/"
                             f"5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{disease.replace(' ', '_')}"
                             f"_complete_metricPerofrmance_emb128.pickle")
        tmp2 = pd.read_pickle(f"PPI_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/"
                              f"5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{disease.replace(' ', '_')}"
                              f"_complete_metricPerofrmance_emb128_DLP.pickle")

        mean_df1 = get_mean_performance_df(tmp1, metrics)
        mean_df2 = get_mean_performance_df(tmp2, metrics)
        mean_df = pd.concat([mean_df1, mean_df2])
        if 'metapath2vec++' in mean_df.index:
            mean_df.drop('metapath2vec++', inplace=True)
        mean_df.index = [method.replace('_', ' ') if method not in methods_nice_name_d else methods_nice_name_d[method]
                         for method in mean_df.index]

        graphsage_perf = pd.read_pickle(f"PPI_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/"
                                        f"graphsage_metrics_emb128.pickle")

        graphsage_perf = best_edgeembed_graphsage(graphsage_perf)

        ap[f"{ppi_scaffold}_{st}"][disease] = pd.concat([mean_df['Average_precision'],
                                                                pd.Series({"GraphSAGE": graphsage_perf['ap']})])
#
# plot_heatmap_performance_values(ap, save_fp=f"PPI_Benchmark_Res{screening}/{ppi_scaffold}/PPI_heatmap_performance",
#                                 save_raw_data=f"PPI_Benchmark_Res{screening}/{ppi_scaffold}/PPI_heatmap_performance.csv",
#                                 include_mean=True, pdf=False)

ap_rnai_reactome = pd.read_csv(f"CellLine_Specific_Benchmark_Res/reactome/ap_emb128_inclPan.csv",
                               header=0, index_col=0)

plot_heatmap_performance_values(ap['STRING_rnai'], ap['STRING_crispr'], ap_rnai_reactome, ap['reactome_rnai'],
                                annotation=['a', 'b', 'c', 'd'],
                                save_fp=f"CellLine_Specific_Benchmark_Res/supp_fig_1_final",
                                save_raw_data=f"CellLine_Specific_Benchmark_Res/supp_fig_1_final.csv",
                                include_mean=False, pdf=False)

