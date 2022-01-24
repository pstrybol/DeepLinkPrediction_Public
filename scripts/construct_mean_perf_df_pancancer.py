from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
from DeepLinkPrediction.utils import plot_heatmap_performance_values
from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np
import glob

BASE_PATH = "/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark"

ppi_scaffold = 'STRING'
disease = 'Pan Cancer'
screening = ''
train_ratio = 80
methods = [f.split('/')[-1].split('_')[0] for f in glob.glob(f"EvalNE_pancancer/method_predictions_{ppi_scaffold}{screening}/*")]
methods = list(filter(('PanCancer').__ne__, methods))
methods.insert(0, 'DLP-DeepWalk')
print(methods)
diseases = ['Bile Duct Cancer', 'Brain Cancer', 'Bladder Cancer', 'Breast Cancer', 'Lung Cancer', 'Prostate Cancer',
            'Skin Cancer']
methods_nice_name_d = {'line-opene': 'LINE', 'n2v-opene': 'node2vec', 'deepwalk-opene': 'DeepWalk',
                       'grarep-opene': 'GraRep', 'original': 'DepMap', 'DLP-hadamard': 'DLP',
                       'DLP-weighted-l2-deepwalk-opene': 'DLP-DeepWalk'}
pan_nw = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                     f"{ppi_scaffold}_Pan_Cancer_dependencies{screening}.csv")
pan_nw_obj = UndirectedInteractionNetwork(pan_nw)

dis2cl = {}
for disease in diseases:
    dis_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}.csv", header=0,
                         index_col=0)
    dis2cl[disease] = set(dis_df.index)

ap_perf = {}
labels = {}
for method in methods:
    print(method)
    ap_perf[method] = {}
    if method != 'DLP-DeepWalk':
        method_fp = [f for f in glob.glob(f"EvalNE_pancancer/method_predictions_{ppi_scaffold}{screening}/*") if method in f][0]
        f_l = sorted(glob.glob(f'{method_fp}/{method}*_test_genes*'))
        f_p = sorted(glob.glob(f'{method_fp}/{method}*_test_preds*'))
    for repeat in range(3):
        if method == 'DLP-DeepWalk':
            tot_df = pd.read_pickle(f"EvalNE_pancancer/method_predictions_{ppi_scaffold}{screening}/"
                                    f"PanCancer_revised_setting_DLP-DeepWalk_{ppi_scaffold}_{repeat}{screening}.pickle")
            tot_df[[0, 1]] = tot_df[[0, 1]].applymap(lambda x: pan_nw_obj.int2gene[x])
            tot_df.columns = ['TestEdges_A', 'TestEdges_B', 'Predictions', 'labels']
            tot_df.sort_values(['TestEdges_A', 'TestEdges_B'], inplace=True)
            labels[f'rep{repeat}'] = tot_df.labels.values
        else:
            test_edges = np.loadtxt(f_l[repeat], delimiter=',', dtype=int)
            test_preds = np.loadtxt(f_p[repeat], delimiter=',', dtype=float)

            tot_df = pd.DataFrame({'TestEdges_A': test_edges[:, 0], 'TestEdges_B': test_edges[:, 1]})
            tot_df = tot_df.applymap(lambda x: pan_nw_obj.int2gene[int(x)])
            tot_df['Predictions'] = test_preds
            tot_df.sort_values(['TestEdges_A', 'TestEdges_B'], inplace=True)
            tot_df['labels'] = labels[f'rep{repeat}']

        for dis, cls in dis2cl.items():
            tmp_df = tot_df[tot_df['TestEdges_A'].apply(lambda x: x in cls)]
            # print(dis, tmp_df.labels.sum())
            if dis in ap_perf[method]:
                ap_perf[method][dis].append(average_precision_score(tmp_df.labels, tmp_df.Predictions))
            else:
                ap_perf[method][dis] = [average_precision_score(tmp_df.labels, tmp_df.Predictions)]


perf_df = pd.DataFrame(ap_perf)
perf_df.to_csv(f"EvalNE_pancancer/method_predictions_{ppi_scaffold}{screening}/ap_per_run_PanCancer.csv",
               header=True, index=True)

perf_df = pd.read_csv(f"EvalNE_pancancer/method_predictions_{ppi_scaffold}{screening}/ap_per_run_PanCancer.csv",
                      header=0, index_col=0)
mean_perf_df = perf_df.applymap(lambda x: np.mean(x)*100).transpose()
mean_perf_df.index = [methods_nice_name_d[d] if d in methods_nice_name_d else d for d in mean_perf_df.index]

plot_heatmap_performance_values(mean_perf_df, include_mean=True, title=f"{ppi_scaffold} Pan Cancer {screening}",
                                save_fp=f"{BASE_PATH}/CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/Pan_Cancer/"
                                        f"PanCancer_revised_heatmap", pdf=True)


