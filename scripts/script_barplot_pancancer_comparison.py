import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle


ppi_scaffold = 'STRING'
screening = ''

perf_df = pd.read_csv(f"EvalNE_pancancer/method_predictions_{ppi_scaffold}{screening}/ap_per_run_PanCancer.csv",
                      header=0, index_col=0)
tmp = perf_df['DLP-DeepWalk'].apply(lambda x: eval(x)).apply(lambda x: np.mean(x))*100
tmp.index = [i.replace(' Cancer', '') for i in tmp.index]
ap_rnai = pd.read_csv(f"CellLine_Specific_Benchmark_Res/{ppi_scaffold}/ap_emb128_inclPan.csv", header=0, index_col=0)
ap_rnai.drop('Pan', axis=1, inplace=True)



plot_ = pd.DataFrame({'Pan Cancer': tmp, 'Cancer Specific': ap_rnai.loc['DLP-DeepWalk']}).melt(ignore_index=False)
plot_['index'] = plot_.index
_, ax = plt.subplots(figsize=(8, 5))
b = sns.barplot(data=plot_, x='index', y='value', hue='variable', palette='colorblind')
b.set_xticklabels(ap_rnai.columns, rotation=30, ha='right')
b.spines['top'].set_visible(False)
b.spines['right'].set_visible(False)
b.spines['bottom'].set_visible(False)
b.legend_.set_title(None)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.set_ylabel("Dependency Specific Average Precision", fontsize=12)
ax.set_xlabel("Cancer Type", fontsize=12)
# plt.show()
plt.savefig(f"CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/barplot_pan_vs_single", dpi=600)
plt.close()


