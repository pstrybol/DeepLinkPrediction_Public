from DeepLinkPrediction.utils import *
import pandas as pd
import pickle
import os
os.chdir('/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark')


ppi_scaffold = 'STRING'
diseases = ['Bile Duct Cancer', 'Prostate Cancer', 'Bladder Cancer', 'Skin Cancer', 'Brain Cancer', 'Breast Cancer',
            'Lung Cancer'] # sorted in ascending number of cell lines
# diseases = ['Bile Duct Cancer', 'Prostate Cancer', 'Bladder Cancer', 'Skin Cancer', 'Brain Cancer', 'Breast Cancer',
#             'Lung Cancer']
npr_ppi = 5
npr_dep = 3
screening = '_crispr'
graphsage = False
metrics = ['tn', 'fp', 'fn', 'tp', 'auroc', 'precision', 'recall', 'fallout', 'miss', 'accuracy', 'f_score',
           'average_precision', 'eval_time']
methods_nice_name_d = {'line-opene': 'LINE', 'n2v-opene': 'node2vec', 'deepwalk-opene': 'DeepWalk',
                       'grarep-opene': 'GraRep', 'DLP-hadamard': 'DLP',
                       'DLP-weighted-l2-deepwalk-opene': 'DLP-DeepWalk'}
# Cell line specific performance ---------------------------------------------------------------------------------------
auroc = {}
ap = {}
f1 = {}
acc = {}
ap_per_run = {}
for disease in diseases:
    print(disease+'\n')
    # Cell Line specific performance
    tmp = pd.read_pickle(f"CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/"
                         f"5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}"
                         f"_cellLinePerformance_emb128_trained80percent.pickle")
    tmp_additional = pd.read_pickle(
        f"CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/"
        f"5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}"
        f"_cellLinePerformance_emb128_deepwalk-opene_pretrainedEMBS_80percent_embs_not_frozen.pickle")

    if graphsage:
        graphsage_perf = pd.read_pickle(f"CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/"
                                   f"graphsage_metrics_emb128.pickle")
        graphsage = {k:[i*100 for i in v] for k, v in graphsage.items()}
        graphsage_perf = best_edgeembed_graphsage(graphsage_perf)

    run_df = pd.concat([tmp.get_pandas_df(metric='average_precision', repeat=i) for i in range(3)], axis=1)
    run_df.columns = ["run0", "run1", "run2"]
    run_df_additional = pd.concat(
        [tmp_additional.get_pandas_df(metric='average_precision', repeat=i) for i in range(3)], axis=1)
    run_df_additional.columns = ["run0", "run1", "run2"]

    # run_ap = pd.concat([run_df, run_df_additional, graphsage_df])
    run_ap = pd.concat([run_df, run_df_additional])

    mean_df = get_mean_performance_df(tmp, metrics)
    mean_df_additional = get_mean_performance_df(tmp_additional, metrics)
    mean_df = pd.concat([mean_df, mean_df_additional])
    mean_df.index = [method.replace('_', ' ') if method not in methods_nice_name_d else methods_nice_name_d[method]
                     for method in mean_df.index]
    if 'metapath2vec++' in mean_df.index:
        mean_df.drop('metapath2vec++', inplace=True)

    if graphsage:
        auroc[disease] = pd.concat([mean_df['Auroc'], pd.Series({"GraphSAGE": graphsage_perf['auc']})])
        ap[disease] = pd.concat([mean_df['Average_precision'], pd.Series({"GraphSAGE": graphsage_perf['ap']})])
        f1[disease] = pd.concat([mean_df['F_score'], pd.Series({"GraphSAGE": graphsage_perf['f1']})])
        acc[disease] = pd.concat([mean_df['Accuracy'], pd.Series({"GraphSAGE": graphsage_perf['acc']})])
        ap_per_run[disease] = pd.concat([run_ap, graphsage_perf['run_ap'] / 100])
    else:
        auroc[disease] = mean_df['Auroc']
        ap[disease] = mean_df['Average_precision']
        f1[disease] = mean_df['F_score']
        acc[disease] = mean_df['Accuracy']
        ap_per_run[disease] = run_ap



# with open(f"CellLine_Specific_Benchmark_Res/{ppi_scaffold}/auroc_across_diseases", 'wb') as handle:
#     pickle.dump(auroc, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(f"CellLine_Specific_Benchmark_Res/{ppi_scaffold}/f1_across_diseases", 'wb') as handle:
#     pickle.dump(f1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/ap_across_diseases", 'wb') as handle:
    pickle.dump(ap_per_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# plot_heatmap_performance_values(auroc,
#                                 save_fp=f'CellLine_Specific_Benchmark_Res/{ppi_scaffold}/auroc_heatmap_emb128',
#                                 save_raw_data=f'CellLine_Specific_Benchmark_Res/{ppi_scaffold}/auroc_heatmap_emb128_raw.csv')
plot_heatmap_performance_values(ap,
                                save_fp=f'CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/ap_heatmap_emb128{screening}',
                                save_raw_data=f'CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/ap_heatmap_emb128_raw{screening}.csv',
                                include_mean=True, pdf=False)

plot_heatmap_performance_values(ap, save_fp=None, save_raw_data=None, include_mean=True)
# plot_heatmap_performance_values(f1,
#                                 save_fp=f'CellLine_Specific_Benchmark_Res/{ppi_scaffold}/f1_heatmap_emb128',
#                                 save_raw_data=f'CellLine_Specific_Benchmark_Res/{ppi_scaffold}/f1_heatmap_emb128_raw.csv')


del tmp, mean_df, auroc, ap, f1

# General performance -------------------------------------------------------------------------------------------------
auroc = {}
ap_general = {}
f1 = {}
for disease in diseases:
    print(disease + '\n')
    # General performance
    tmp = pd.read_pickle(f"General_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/"
                         f"5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{disease.replace(' ', '_')}"
                         f"_emb128.pickle")

    tmp_additional_2 = pd.read_pickle(f"General_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/"
                                      f"5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{disease.replace(' ','_')}"
                                      f"_complete_metricPerofrmance_emb128_deepwalk-opene_pretrainedEMBS_"
                                      f"80percent_embs_not_frozen.pickle")

    tmp_additional_3 = pd.read_pickle(f"General_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/"
                         f"5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{disease.replace(' ', '_')}"
                         f"_emb128_baselines.pickle")

    graphsage_perf = pd.read_pickle(f"General_Benchmark_Res{screening}/{ppi_scaffold}/"
                                    f"{disease.replace(' ', '_')}/graphsage_metrics_emb128.pickle")

    graphsage_perf = best_edgeembed_graphsage(graphsage_perf)

    mean_df = get_mean_performance_df(tmp, metrics)
    mean_df_add_2 = get_mean_performance_df(tmp_additional_2, metrics)
    mean_df_add_3 = get_mean_performance_df(tmp_additional_3, metrics)
    mean_df = pd.concat([mean_df, mean_df_add_2, mean_df_add_3])
    mean_df.drop('metapath2vec++', inplace=True)

    mean_df.index = [method.replace('_', '-') if method not in methods_nice_name_d else methods_nice_name_d[method]
                     for method in mean_df.index]

    auroc[disease] = pd.concat([mean_df['Auroc'], pd.Series({"GraphSAGE": graphsage_perf['auc']})])
    ap_general[disease] = pd.concat([mean_df['Average_precision'], pd.Series({"GraphSAGE": graphsage_perf['ap']})])
    f1[disease] = pd.concat([mean_df['F_score'], pd.Series({"GraphSAGE": graphsage_perf['f1']})])

# plot_heatmap_performance_values(auroc,
#                                 save_fp=f'General_Benchmark_Res/{ppi_scaffold}/auroc_heatmap_emb128',
#                                 save_raw_data=f'General_Benchmark_Res/{ppi_scaffold}/auroc_heatmap_emb128_raw.csv')
# plot_heatmap_performance_values(ap_general,
#                                 save_fp=f'General_Benchmark_Res/{ppi_scaffold}/ap_heatmap_emb128',
#                                 save_raw_data=f'General_Benchmark_Res/{ppi_scaffold}/ap_heatmap_emb128_raw.csv')
# plot_heatmap_performance_values(f1,
#                                 save_fp=f'General_Benchmark_Res/{ppi_scaffold}/f1_heatmap_emb128',
#                                 save_raw_data=f'General_Benchmark_Res/{ppi_scaffold}/f1_heatmap_emb128_raw.csv')

plot_heatmap_performance_values(ap_general, ap, annotation=["a", "b"],
                                save_fp=f'CellLine_Specific_Benchmark_Res/{ppi_scaffold}/ap_heatmap_emb12_cellsys_revised_reactome',
                                include_mean=True, pdf=True)

plot_heatmap_performance_values(ap_general, ap, annotation=["a", "b"],
                                save_fp=None, include_mean=True)

del auroc, ap, f1