from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
from DeepLinkPrediction.utils import read_h5py, read_ppi_scaffold, construct_EvalNE_splits, eval_other, eval_baselines, evaluate_dw
from sklearn.metrics import average_precision_score
from evalne.evaluation import edge_embeddings
from evalne.evaluation.score import Scoresheet
from DeepLinkPrediction import main_DLP
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import pickle
import glob
import os
import re
# Load the network and train/test edges
diseases = ['Bile Duct Cancer', 'Prostate Cancer', 'Bladder Cancer', 'Skin Cancer', 'Brain Cancer', 'Breast Cancer',
            'Lung Cancer']
crispr_thresholds = {'Bile Duct Cancer': -1.822879, 'Prostate Cancer': -2.02022, 'Bladder Cancer': -2.029239,
                     'Skin Cancer': -2.04098, 'Brain Cancer': -2.02018, 'Breast Cancer': -1.92688,
                     'Lung Cancer': -1.99309}

ppi_scaffold = "reactome"
screening = '_crispr'
pos_thresh = ''
npr_ppi = 5
npr_dep = 3
emb_dim = 128

BASE_PATH = "/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark"
print(ppi_scaffold, screening, '\n')

pan_nw = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                     f"{ppi_scaffold}_Pan_Cancer_dependencies{screening}.csv")
pan_nw_obj = UndirectedInteractionNetwork(pan_nw)

save_fp = f"{BASE_PATH}/CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/Pan_Cancer"
if not os.path.isdir(save_fp):
    os.makedirs(save_fp)

ppi_obj = read_ppi_scaffold(ppi_scaffold, f"{BASE_PATH}/ppi_network_scaffolds/")

ppiINT2panINT = {v: pan_nw_obj.gene2int[k] for k, v in ppi_obj.gene2int.items()}

# ap_performance = {}
scoresheet = Scoresheet(tr_te='test')
for repeat in range(3):
    # repeat = 0
    print('Repetition {} of experiment'.format(repeat))

    # ap_performance[repeat] = {}
    X_train_, X_val_, X_test_, Y_train_, Y_val_, Y_test_ = ppi_obj.getTrainTestData(neg_pos_ratio=npr_ppi,
                                                                                    train_ratio=0.8,
                                                                                    train_validation_ratio=0.8,
                                                                                    return_summary=False)

    train_and_val_edges_to_stack = []
    train_and_val_labels_DEP_to_stack = []
    tot_shape = 0

    only_train_edges_DEP_to_stack = []
    only_train_labels_DEP_to_stack = []

    only_val_edges_DEP_to_stack = []
    only_val_labels_DEP_to_stack = []
    test_sets_per_disease_to_combine = {}
    total_test_sets = {}

    for dis in diseases:
        print(dis)
        if screening == '_crispr':
            pos_thresh = f"_pos{str(crispr_thresholds[dis]).replace('.', '_')}"

        heterogeneous_network = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                                            f"{ppi_scaffold}_{dis.replace(' ', '_')}_dependencies{screening}{pos_thresh}.csv")
        nw_obj = UndirectedInteractionNetwork(heterogeneous_network)
        disINT2panINT = {v: pan_nw_obj.gene2int[k] for k, v in nw_obj.gene2int.items()}
        cls = set(nw_obj.node_names) - set(ppi_obj.node_names)
        cls_int_ = [nw_obj.gene2int[i] for i in cls]
        cls_int = [disINT2panINT[i] for i in cls_int_]
        all_file_loc = glob.glob(f"{BASE_PATH}/LP_train_test_splits{screening}/"
                                 f"{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}{pos_thresh}/"
                                 f"{dis.replace(' ', '_')}/*")

        # --------------------------------- TRAIN SET ----------------------------------------------------------------
        train_and_val_edges_df = pd.DataFrame(read_h5py(list(filter(re.compile(rf'(.*/trE_{repeat}.hdf5)').match,
                                                                 all_file_loc))[0]),
                                           columns=['GeneA', 'GeneB']).applymap(lambda x: disINT2panINT[x])
        train_and_val_edges_df_idx = train_and_val_edges_df.applymap(lambda x: x in cls_int)
        train_and_val_edges_DEP = train_and_val_edges_df[train_and_val_edges_df_idx.GeneA | train_and_val_edges_df_idx.GeneB]
        tot_shape += train_and_val_edges_DEP.shape[0]

        train_and_val_labels = read_h5py(list(filter(re.compile(rf'(.*/label_trE_{repeat}.hdf5)').match, all_file_loc))[0])
        train_and_val_labels_DEP = train_and_val_labels[train_and_val_edges_DEP.index]

        train_and_val_edges_to_stack.append(train_and_val_edges_DEP)
        train_and_val_labels_DEP_to_stack.append(train_and_val_labels_DEP)

        # --------------------------------- TEST SET ----------------------------------------------------------------
        total_test_set = pd.DataFrame(read_h5py(f"{BASE_PATH}/LP_train_test_splits{screening}/"
                                                f"{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}"
                                                f"{pos_thresh}/{dis.replace(' ','_')}/"
                                                f"complete_predset{'_without_cl2cl' if screening != '' else ''}"
                                                f".hdf5")).applymap(lambda x: disINT2panINT[x])

        test_edges_to_combine = pd.DataFrame(read_h5py(list(filter(re.compile(rf'(.*/test_all_cls_{repeat}.hdf5)').match,
                                                        all_file_loc))[0])).applymap(lambda x: disINT2panINT[x])
        test_labels_to_combine = read_h5py(
            list(filter(re.compile(rf'(.*/label_test_all_cls_{repeat}.hdf5)').match, all_file_loc))[0])

        test_sets_per_disease_to_combine[dis] = tuple((test_edges_to_combine.values, test_labels_to_combine))

        total_test_sets[dis] = total_test_set

        # --------------------------------- VALIDATION SET ----------------------------------------------------------------
        only_train_edges_df = pd.DataFrame(read_h5py(list(filter(re.compile(rf'(.*/trE_val_{repeat}.hdf5)').match,
                                                                 all_file_loc))[0]),
                                           columns=['GeneA', 'GeneB']).applymap(lambda x: disINT2panINT[x])
        only_train_edges_df_idx = only_train_edges_df.applymap(lambda x: x in cls_int)
        only_train_edges_DEP = only_train_edges_df[only_train_edges_df_idx.GeneA | only_train_edges_df_idx.GeneB]

        only_train_labels_val = read_h5py(
            list(filter(re.compile(rf'(.*/label_trE_val_{repeat}.hdf5)').match, all_file_loc))[0])
        only_train_labels_DEP = only_train_labels_val[only_train_edges_DEP.index]

        only_val_edges_df = pd.DataFrame(read_h5py(list(filter(re.compile(rf'(.*/teE_val_{repeat}.hdf5)').match,
                                                               all_file_loc))[0]),
                                         columns=['GeneA', 'GeneB']).applymap(lambda x: disINT2panINT[x])
        only_val_edges_df_idx = only_val_edges_df.applymap(lambda x: x in cls_int)
        only_val_edges_DEP = only_val_edges_df[only_val_edges_df_idx.GeneA | only_val_edges_df_idx.GeneB]

        only_val_labels = read_h5py(
            list(filter(re.compile(rf'(.*/label_teE_val_{repeat}.hdf5)').match, all_file_loc))[0])
        only_val_labels_DEP = only_val_labels[only_val_edges_DEP.index]

        only_train_edges_DEP_to_stack.append(only_train_edges_DEP)
        only_train_labels_DEP_to_stack.append(only_train_labels_DEP)

        only_val_edges_DEP_to_stack.append(only_val_edges_DEP)
        only_val_labels_DEP_to_stack.append(only_val_labels_DEP)

    # TRAINING
    train_and_val_edges_DEP_df = pd.concat(train_and_val_edges_to_stack, axis=0)
    assert train_and_val_edges_DEP_df.shape[0] == tot_shape, f"Shape mismatch {train_and_val_edges_DEP_df.shape[0]} vs {tot_shape}"
    pan_cancer_train_edges = np.vstack((X_train_, X_val_, train_and_val_edges_DEP_df.values))
    assert np.unique(pan_cancer_train_edges.ravel()).shape[0] == pan_nw_obj.nodes.shape[0], "Shape mismatch"

    pan_cancer_train_labels = np.hstack((Y_train_, Y_val_, np.hstack(train_and_val_labels_DEP_to_stack)))
    assert pan_cancer_train_labels.shape[0] == pan_cancer_train_edges.shape[0], "Shape mismatch"

    # VALIDATION
    only_train_edges_DEP_df = pd.concat(only_train_edges_DEP_to_stack, axis=0)
    pan_cancer_val_edges = np.vstack((X_train_, only_train_edges_DEP_df))
    pan_cancer_val_labels = np.hstack((Y_train_, np.hstack(only_train_labels_DEP_to_stack)))

    only_val_edges_DEP_df = pd.concat(only_val_edges_DEP_to_stack, axis=0)
    pan_cancer_val_test_edges = np.vstack((X_val_, only_val_edges_DEP_df))
    pan_cancer_val_test_labels = np.hstack((Y_val_, np.hstack((only_val_labels_DEP_to_stack))))

    # COMBINE TRAINING AND TESTING LEAVE VALIDATION ALONE
    test_edges_to_combine_total = np.vstack([i[0] for i in test_sets_per_disease_to_combine.values()])
    test_labels_to_combine = np.hstack([i[1] for i in test_sets_per_disease_to_combine.values()])

    pan_cancer_100train_edges = np.vstack((pan_cancer_train_edges, test_edges_to_combine_total))
    pan_cancer_100train_labels = np.hstack((pan_cancer_train_labels, test_labels_to_combine))

    # TESTING
    test_total_edges = np.vstack(list(total_test_sets.values()))
    test_total_labels = np.array([random.randint(0, 1) for _ in range(test_total_edges.shape[0])])

    nee = construct_EvalNE_splits(pan_cancer_100train_edges, pan_cancer_100train_labels, pan_cancer_val_edges,
                                  pan_cancer_val_labels, test_total_edges, test_total_labels,
                                  pan_cancer_val_test_edges, pan_cancer_val_test_labels,
                                  ppi_scaffold=ppi_scaffold, repeat=repeat, dim=emb_dim)

    eval_other(nee=nee, scoresheet=scoresheet, npr_ppi=npr_ppi, npr_dep=npr_dep, disease="Pan Cancer", train_ratio=100,
               BASE_PATH=BASE_PATH, save_preds=f"{BASE_PATH}/EvalNE_pancancer_target_prediction/{ppi_scaffold}{screening}", save_embs=True, emb_size=emb_dim,
               ppi_scaffold=ppi_scaffold, screening=screening, pos_thresh='')

    eval_baselines(nee=nee, directed=False, scoresheet=scoresheet, save_preds=f"{BASE_PATH}/EvalNE_pancancer_target_prediction/{ppi_scaffold}{screening}")

    # evaluate_dw(ppi_scaffold=ppi_scaffold, disease="Pan Cancer", screening=screening, train_edges=pan_cancer_100train_edges,
    #             train_labels=pan_cancer_100train_labels, test_edges=test_total_edges,
    #             test_labels=test_total_labels, train_edges_val=pan_cancer_val_edges,
    #             train_labels_val=pan_cancer_val_labels, test_edges_val=pan_cancer_val_test_edges,
    #             test_labels_val=pan_cancer_val_test_labels, repeat=repeat, npr_ppi=5, npr_dep=3,
    #             pos_thresh="", save_embs=True,
    #             save_preds=f"{BASE_PATH}/EvalNE_pancancer_target_prediction/{ppi_scaffold}{screening}", train_ratio=100)


    print("Starting training...\n")
    dl_obj = main_DLP.main(tr_e=pan_cancer_train_edges, tr_e_labels=pan_cancer_train_labels, inputgraph=pan_nw_obj,
                           merge_method='weighted-l2', predifined_embs=f"deepwalk-opene_Pan_Cancer_{ppi_scaffold}_"
                                                                    f"embsize{emb_dim}_100percent{screening}" \
                                                                    f"_nprPPI{npr_ppi}_nprDEP{npr_dep}/emb_deepwalk-opene_{repeat}.tmp")


    tmp_df = pd.DataFrame(test_total_edges)
    tmp_df[f'preds_rep{repeat}'] = dl_obj.predict_proba(test_total_edges)
    tmp_df[f'labels'] = test_total_labels
    pd.to_pickle(tmp_df, f"{BASE_PATH}/EvalNE_pancancer_target_prediction/{ppi_scaffold}{screening}/"
                         f"PanCancer_revised_setting_DLP-DeepWalk_predictions_{ppi_scaffold}_{repeat}.pickle")

# -------------------------------------------------------- READ IN RESULTS -------------------------------------------
# perf = pd.DataFrame(pd.read_pickle(f"{save_fp}/pancancer_performance_separate_cts.pickle"))*100
#
# h = sns.heatmap(perf, cmap="RdYlBu", vmin=0, vmax=100, square=False, linewidth=0.3, annot=True)
# # h.set_xticklabels(perf_df.columns, rotation=30, ha='right')
# plt.show()