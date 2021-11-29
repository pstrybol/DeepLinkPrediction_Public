from evalne.evaluation.score import Scoresheet
from DeepLinkPrediction.utils import *
from DeepLinkPrediction import main_DLP
import pandas as pd
import numpy as np
import pickle
import random
import glob
import re
import os
disease = 'Lung Cancer'
ppi_scaffold = "STRING"
screening = ''
pos_thresh = ''
npr_ppi = 5
npr_dep = 3
emb_dim = 128
assert os.getcwd().split('/')[-1] == "EvalNE_transductive", "Wrong working directory"
BASE_PATH = '/'.join(os.getcwd().split('/')[:-1])

original_dis_df = pd.read_csv(
        f"{BASE_PATH}/depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}{screening}"
        f"{pos_thresh}.csv",
        header=0, index_col=0)
original_pos = extract_pos_dict_at_threshold(original_dis_df, threshold=-1.5)

ppi_obj = read_ppi_scaffold(ppi_scaffold, f"{BASE_PATH}/ppi_network_scaffolds/")

all_file_loc = glob.glob(f"{BASE_PATH}/LP_train_test_splits{screening}/"
                         f"transductive20_{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}{pos_thresh}/"
                         f"{disease.replace(' ', '_')}/*")
heterogeneous_network = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                                    f"transductive20_{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies{screening}.csv")
nw_obj = UndirectedInteractionNetwork(heterogeneous_network)

dis_df = pd.read_csv(
        f"{BASE_PATH}/depmap_specific_cancer_df/transductive20_{ppi_scaffold}_{disease.replace(' ', '_')}{screening}"
        f"{pos_thresh}.csv", header=0, index_col=0)

dropout = set(original_dis_df.columns) - set(dis_df.columns)
test_l = []
test_labels = []
for gene in dropout:
    test_l.append(pd.DataFrame([[nw_obj.gene2int[gene], nw_obj.gene2int[b]] for b in dis_df.index],
                               columns=['GeneA', 'GeneB']))
    test_labels.append([1 if gene in original_pos[b] else 0 for b in dis_df.index])

test_df = pd.concat(test_l)
test_edges = test_df.values
test_labels = np.hstack(test_labels)
test_df['label'] = test_labels

scoresheet = Scoresheet(tr_te='test')
for repeat in range(3):
    print(repeat)
    train_edges = read_h5py(list(filter(re.compile(rf'(.*/trE_{repeat}.hdf5)').match, all_file_loc))[0])
    train_labels = read_h5py(list(filter(re.compile(rf'(.*/label_trE_{repeat}.hdf5)').match, all_file_loc))[0])

    train_edges_val = read_h5py(list(filter(re.compile(rf'(.*/trE_val_{repeat}.hdf5)').match, all_file_loc))[0])
    train_labels_val = read_h5py(list(filter(re.compile(rf'(.*/label_trE_val_{repeat}.hdf5)').match, all_file_loc))[0])
    test_edges_val = read_h5py(list(filter(re.compile(rf'(.*/teE_val_{repeat}.hdf5)').match, all_file_loc))[0])
    test_labels_val = read_h5py(list(filter(re.compile(rf'(.*/label_teE_val_{repeat}.hdf5)').match, all_file_loc))[0])

    nee = construct_EvalNE_splits(train_edges, train_labels, train_edges_val,
                                  train_labels_val, test_edges, test_labels,
                                  test_edges_val, test_labels_val,
                                  ppi_scaffold=ppi_scaffold, repeat=repeat, dim=emb_dim)

    # eval_AROPE(nee=nee, scoresheet=scoresheet, BASE_PATH=BASE_PATH,
    #            save_preds=f"{BASE_PATH}/transductive_setting_20perc/{ppi_scaffold}{screening}/{disease}")

    # eval_baselines(nee=nee, directed=False, scoresheet=scoresheet,
    #                save_preds=f"{BASE_PATH}/transductive_setting_20perc/{ppi_scaffold}{screening}/{disease}")

    evaluate_dw(ppi_scaffold=ppi_scaffold, disease=disease, screening=screening, train_edges=train_edges,
                train_labels=train_labels, test_edges=test_edges,
                test_labels=test_labels, train_edges_val=train_edges_val,
                train_labels_val=train_labels_val, test_edges_val=test_edges_val,
                test_labels_val=test_labels_val, repeat=repeat, npr_ppi=5, npr_dep=3,
                pos_thresh=pos_thresh, save_embs=True,
                save_preds=f"{BASE_PATH}/transductive_setting_20perc/{ppi_scaffold}{screening}/{disease}")

    dl_obj = main_DLP.main(tr_e=train_edges, tr_e_labels=train_labels, inputgraph=nw_obj,
                           merge_method='hadamard', predifined_embs=None)
    test_df[f'predictions_rep{repeat}'] = dl_obj.predict_proba(test_edges)

    # dl_obj_dw = main_DLP.main(tr_e=train_edges, tr_e_labels=train_labels, inputgraph=nw_obj,
    #                        merge_method='weighted-l2',
    #                        predifined_embs=f"deepwalk-opene_{disease.replace(' ', '_')}_{ppi_scaffold}_"
    #                                        f"embsize128_80percent{screening}{pos_thresh}" \
    #                                        f"_nprPPI{npr_ppi}_nprDEP{npr_dep}/emb_deepwalk-opene_{repeat}.tmp")
    # test_df[f'predictions_rep{repeat}'] = dl_obj_dw.predict_proba(test_edges)



pd.to_pickle(test_df, f"{BASE_PATH}/transductive_setting_20perc/{ppi_scaffold}{screening}/{disease}/"
                      f"DLP-hadamard_predictions.pickle")
