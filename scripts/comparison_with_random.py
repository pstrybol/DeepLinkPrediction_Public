from DeepLinkPrediction.utils import *
import pickle
import os
os.chdir('/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark')

with open('drug_sensitivity_data/100percent_final/total_target_retrieval_all_diseases_no_baselines_dict.pickle', 'rb') as handle:
    total_sensitive_targets_retrieved = pickle.load(handle)

drug_sens = pd.read_csv('depmap_data/processed_primary-screen-replicate-logfold.csv', header=0, index_col=0)

methods_nice_name_d = {'line-opene': 'LINE', 'n2v-opene': 'node2vec', 'deepwalk-opene': 'DeepWalk',
                       'grarep-opene': 'GraRep', 'original': 'DepMap', 'DLP-hadamard': 'DLP',
                       'DLP-weighted-l2-deepwalk-opene': 'DLP-DeepWalk'}
methods = ['DLP-weighted-l2-deepwalk-opene', 'grarep-opene', 'DLP-hadamard', 'deepwalk-opene', 'AROPE', 'original']
diseases = ["Lung Cancer"]
ppi_scaffold = "STRING"
drug_thresh = -2
N_PERMUTATIONS = 10_000
test = "hypergeom"

random_significantly_better = {}
random_significantly_worse = {}
better_than_random_setting1 = {}
pval_method_better_random = {}
for_dist_plot = {}
d_ = {}
times_significant = {}
for disease in diseases:
    d_[disease] = {}
    random_significantly_better[disease] = {}
    random_significantly_worse[disease] = {}
    better_than_random_setting1[disease] = {}
    pval_method_better_random[disease] = {}
    for_dist_plot[disease] = []

    ppi_scaffold_object = read_ppi_scaffold(ppi_scaffold, 'ppi_network_scaffolds/')
    degree_df = ppi_scaffold_object.getDegreeDF(set_index=True)
    dis_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}.csv", header=0,
                         index_col=0)

    common_cls = set(dis_df.index) & set(drug_sens.index)
    subset_drug_sens = drug_sens.loc[common_cls]

    total_sensitive_targets = {}
    for i, cl in enumerate(common_cls):
        print(f"{i+1}/{len(common_cls)} - {cl}")
        random_significantly_better[disease][cl] = {}
        random_significantly_worse[disease][cl] = {}
        better_than_random_setting1[disease][cl] = {}

        with open(f"drug_sensitivity_data/targets_per_cell_line/Lung Cancer/{cl}_targets_min2.txt", "r") as f:
            total_sensitive_targets[cl] = set([i.strip('\n') for i in f.readlines()])

    # Random uniform distribution
    if test == "hypergeom":
        for cl in common_cls:
            options = dis_df.columns.tolist()
            hypergeom_without_degree = hypergeom(M=len(options), n=len(total_sensitive_targets[cl]), N=100)
            av_without_degree = hypergeom_without_degree.mean()

            for method in methods:
                no_targets_found = len(total_sensitive_targets_retrieved[disease][cl][method])
                random_significantly_better[disease][cl][method] = hypergeom_without_degree.sf(no_targets_found)
                random_significantly_worse[disease][cl][method] = hypergeom_without_degree.cdf(no_targets_found)
                if no_targets_found > av_without_degree:
                    better_than_random_setting1[disease][cl][method] = no_targets_found - av_without_degree
                else:
                    better_than_random_setting1[disease][cl][method] = 0

    # Random degree-based distribution
    elif test == "permutation":
        if len(common_cls) == 39:
            k = 13
            i_ = int(39 / k)
        elif len(common_cls) == 88:
            k = 11
            i_ = int(88 / k)
        else:
            k = len(common_cls)
            i_ = 1
        common_cls = list(common_cls)
        for i in range(i_):
            print(f"{i + 1}/{i_} - {disease}")
            options = dis_df.columns.tolist()
            probs = degree_df.loc[options].Count.values / degree_df.loc[options].Count.sum()
            backend = 'multiprocessing'
            path_func = delayed(run_permutations)
            random_significantly_better_, random_significantly_worse_,\
            better_than_random_ = zip(*Parallel(n_jobs=k, verbose=True, backend=backend)(
                path_func(N_PERMUTATIONS, options, 100, total_sensitive_targets[cl], methods,
                          total_sensitive_targets_retrieved, cl, disease, weights=probs, seed=n)
                for n, cl in enumerate(common_cls[i * k:(i * k) + k])))

            random_significantly_better[disease].update({cl: rttr
                                                         for cl, rttr in zip(common_cls[i*k:(i*k)+k],
                                                                             random_significantly_better_)})
            random_significantly_worse[disease].update({cl: rttr
                                                         for cl, rttr in zip(common_cls[i * k:(i * k) + k],
                                                                             random_significantly_worse_)})
            better_than_random_setting1[disease].update({cl: btr for cl, btr in zip(common_cls[i * k:(i * k) + k],
                                                                                    better_than_random_)})

    if test == "hypergeom":
        save_fp = "drug_sensitivity_data/100percent_final/uniform_random_sampling_newcmap"
        save_raw_data = "drug_sensitivity_data/100percent_final/uniform_random_sampling_newcmap_raw.csv"
    else:
        save_fp = "drug_sensitivity_data/100percent_final/degreebased_random_sampling_newcmap"
        save_raw_data = "drug_sensitivity_data/100percent_final/degreebased_random_sampling_newcmap_raw.csv"

    times_significant[test] = calculate_stacked_barplot(random_significantly_better, random_significantly_worse,
                                                        common_cls, disease, save_raw_data=save_raw_data,
                                                        methods_nice_name_d=methods_nice_name_d)


times_significant = {}
times_significant["degree_based"] = \
    pd.read_csv(f"drug_sensitivity_data_{ppi_scaffold}/100percent_final/degreebased_random_sampling_newcmap_raw.csv",
                header=0, index_col=0)

times_significant["hypergeom"] = \
    pd.read_csv(f"drug_sensitivity_data_{ppi_scaffold}/100percent_final/uniform_random_sampling_newcmap_raw.csv",
                header=0, index_col=0)

times_significant["degree_based"].drop("significantly worse", axis=1, inplace=True)
times_significant["degree_based"].columns = ['better, yet not significantly',
                                             'significantly better']

times_significant["hypergeom"].drop("significantly worse", axis=1, inplace=True)
times_significant["hypergeom"].columns = ['better, yet not significantly',
                                          'significantly better']

fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.5))
annot = ["a", "b"]
legends = [False, True]
for i, val in enumerate(list(times_significant.values())[::-1]):
    plot_stacked_barplot(val, save_fp=None, ax=axs[i], annotation=annot[i], pdf=True, legend=legends[i])
# plt.show()
save_fp = f"drug_sensitivity_data_{ppi_scaffold}/100percent_final/"\
          f"comparison_with_random_figure4_revised.pdf"
plt.savefig(save_fp, dpi=600)
