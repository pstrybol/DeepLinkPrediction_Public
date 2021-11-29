from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, accuracy_score
from evalne.evaluation import edge_embeddings
from DeepLinkPrediction.utils import read_h5py
import numpy as np
import pandas as pd
import argparse
import random
import pickle
import glob
import re
import os
os.chdir("/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark")

parser = argparse.ArgumentParser()
parser.add_argument('--ppi_scaffold', required=True, type=str)
parser.add_argument('--disease', required=True, type=str)
parser.add_argument('--screening', required=True, type=str)
parser.add_argument('--pos_thresh', required=False, type=str)
parser.add_argument('--general_performance', required=False, action='store_true')
parser.add_argument('--target_prediction', required=False, action='store_true')
parser.add_argument('--total_performance', required=False, action='store_true')
# parser.add_argument('--npr_ppi', required=True, type=int)
# parser.add_argument('--npr_dep', required=True, type=int)
# parser.add_argument('--emb_dim', required=True, type=int)
args = parser.parse_args()

BASE_PATH = "/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark"
ppi_scaffold = args.ppi_scaffold
disease = args.disease
screening = '' if args.screening == 'rnai' else '_crispr'
pos_thresh = f"_pos{args.pos_thresh.replace('.', '_')}" if args.pos_thresh else ""
npr_ppi = 5
npr_dep = 3
edge_embed_method = ['average', 'hadamard', 'weighted_l1', 'weighted_l2']
EMB_PATH = f"GraphSAGE_logdir/unsup-{disease.replace(' ', '_')}/graphsage_seq_small_0.000010/"
all_file_loc = glob.glob(f"LP_train_test_splits{screening}/"
                         f"{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}/{disease.replace(' ','_')}/*")

dis_embs_values = np.load(EMB_PATH + "val.npy")
assert dis_embs_values.shape[1] == 128, f"Wrong number of embeddings dimensions, {dis_embs_values.shape}"
dis_embs_nodes = np.loadtxt(EMB_PATH + "val.txt").astype(int)
dis_embs = {str(v): dis_embs_values[i] for i, v in enumerate(dis_embs_nodes)}

if args.total_performance:
    metrics_general = {}
    metrics_dependencies = {}
else:
    metrics = {}

for edge_method in edge_embed_method:
    if args.total_performance:
        metrics_general[f"ap_{edge_method}"] = []
        metrics_general[f"f1_{edge_method}"] = []
        metrics_general[f"acc_{edge_method}"] = []
        metrics_general[f"auc_{edge_method}"] = []

        metrics_dependencies[f"ap_{edge_method}"] = []
        metrics_dependencies[f"f1_{edge_method}"] = []
        metrics_dependencies[f"acc_{edge_method}"] = []
        metrics_dependencies[f"auc_{edge_method}"] = []
    else:
        metrics[f"ap_{edge_method}"] = []
        metrics[f"f1_{edge_method}"] = []
        metrics[f"acc_{edge_method}"] = []
        metrics[f"auc_{edge_method}"] = []
    for repeat in range(3):
        train_edges = read_h5py(list(filter(re.compile(rf'(.*/trE_{repeat}.hdf5)').match, all_file_loc))[0])
        train_labels = read_h5py(list(filter(re.compile(rf'(.*/label_trE_{repeat}.hdf5)').match, all_file_loc))[0])

        # If we want to calculate performance
        if args.general_performance:
            print("\nGENERAL PEROFRMANCE\n")
            test_edges = read_h5py(list(filter(re.compile(rf'(.*/teE_{repeat}.hdf5)').match, all_file_loc))[0])
            test_labels = read_h5py(list(filter(re.compile(rf'(.*/label_teE_{repeat}.hdf5)').match, all_file_loc))[0])
        elif args.target_prediction:
            print("\nTARGET PREDICTION\n")
            test_edges = read_h5py(f"{BASE_PATH}/LP_train_test_splits{screening}/"
                                f"{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}{pos_thresh}/{disease.replace(' ', '_')}/"
                                f"complete_predset_without_cl2cl.hdf5")
            assert pd.DataFrame(test_edges).shape == pd.DataFrame(test_edges).drop_duplicates().shape

            # Note: random labels are generated since we are predicting on the whole dataset, not used for evaluating performance
            test_labels = np.array([random.randint(0, 1) for _ in range(test_edges.shape[0])])

        elif args.total_performance:
            print("\nTOAL PERFORMANCE: GENERAL + DEPENDENCIES\n")
            test_edges_gen = read_h5py(list(filter(re.compile(rf'(.*/teE_{repeat}.hdf5)').match, all_file_loc))[0])
            test_labels_gen = read_h5py(list(filter(re.compile(rf'(.*/label_teE_{repeat}.hdf5)').match, all_file_loc))[0])

            test_edges_dep = read_h5py(list(filter(re.compile(rf'(.*/test_all_cls_{repeat}.hdf5)').match, all_file_loc))[0])
            test_labels_dep = read_h5py(
                list(filter(re.compile(rf'(.*/label_test_all_cls_{repeat}.hdf5)').match, all_file_loc))[0])
        else:
            print("\nDEPENDENCY SPECIFIC PEROFRMANCE\n")
            test_edges = read_h5py(list(filter(re.compile(rf'(.*/test_all_cls_{repeat}.hdf5)').match, all_file_loc))[0])
            test_labels = read_h5py(list(filter(re.compile(rf'(.*/label_test_all_cls_{repeat}.hdf5)').match, all_file_loc))[0])

        print("Generating edge embeddings ...\n")
        func = getattr(edge_embeddings, str(edge_method))
        tr_edge_embeds = func(dis_embs, train_edges)
        if args.total_performance:
            te_edge_embeds = {}
            te_edge_embeds['general'] = func(dis_embs, test_edges_gen)
            te_edge_embeds['dependencies'] = func(dis_embs, test_edges_dep)
        else:
            te_edge_embeds = func(dis_embs, test_edges)

        print("Fitting the LogReg model ...\n")
        lp_model = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', scoring='roc_auc',
                                        solver='lbfgs', max_iter=100, verbose=0)
        lp_model.fit(tr_edge_embeds, train_labels)

        print("Predicting test edges ...\n")
        # train_pred = lp_model.predict_proba(tr_edge_embeds)[:, 1]
        if args.total_performance:
            test_pred = {}
            test_pred_bin = {}
            for k, v in te_edge_embeds.items():
                test_pred[k] = lp_model.predict_proba(v)[:, 1]
                test_pred_bin[k] = lp_model.predict(v)
        else:
            test_pred = lp_model.predict_proba(te_edge_embeds)[:, 1]
            test_pred_bin = lp_model.predict(te_edge_embeds)
        
        if args.target_prediction:
            save_fp = f"{BASE_PATH}/" \
                      f"onlyDEPALL_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_" \
                      f"emb{dis_embs_values.shape[1]}{screening}{pos_thresh}/" \
                      f"{disease.replace(' ', '_')}_100percent/"
            
            try:
                os.makedirs(save_fp + f"GraphSAGE-{edge_method}" + '_ee_e2e_test_preds')
                print("folder '{}' created ".format(save_fp + f"GraphSAGE-{edge_method}" + '_ee_e2e_test_preds'))
            except FileExistsError:
                print("folder {} already exists".format(save_fp + f"GraphSAGE-{edge_method}" + '_ee_e2e_test_preds'))
            finally:
                np.savetxt(
                    fname=save_fp + f"GraphSAGE-{edge_method}" + '_ee_e2e_test_preds/' + "GraphSAGE" + '_' + ppi_scaffold + '_test_preds_' + str(
                        repeat) + '.txt', X=test_pred, delimiter=',',
                    fmt='%1.6f')
                np.savetxt(
                    fname=save_fp + f"GraphSAGE-{edge_method}" + '_ee_e2e_test_preds/' + "GraphSAGE" + '_' + ppi_scaffold + '_test_genes_' + str(
                        repeat) + '.txt', X=test_edges, delimiter=',',
                    fmt='%i')

        if args.total_performance:
            metrics_general[f"ap_{edge_method}"].append(average_precision_score(test_labels_gen, test_pred['general']))
            metrics_general[f"f1_{edge_method}"].append(f1_score(test_labels_gen, test_pred_bin['general']))
            metrics_general[f"acc_{edge_method}"].append(accuracy_score(test_labels_gen, test_pred_bin['general']))
            metrics_general[f"auc_{edge_method}"].append(roc_auc_score(test_labels_gen, test_pred['general']))

            metrics_dependencies[f"ap_{edge_method}"].append(average_precision_score(test_labels_dep, test_pred['dependencies']))
            metrics_dependencies[f"f1_{edge_method}"].append(f1_score(test_labels_dep, test_pred_bin['dependencies']))
            metrics_dependencies[f"acc_{edge_method}"].append(accuracy_score(test_labels_dep, test_pred_bin['dependencies']))
            metrics_dependencies[f"auc_{edge_method}"].append(roc_auc_score(test_labels_dep, test_pred['dependencies']))
        else:
            metrics[f"ap_{edge_method}"].append(average_precision_score(test_labels, test_pred))
            metrics[f"f1_{edge_method}"].append(f1_score(test_labels, test_pred_bin))
            metrics[f"acc_{edge_method}"].append(accuracy_score(test_labels, test_pred_bin))
            metrics[f"auc_{edge_method}"].append(roc_auc_score(test_labels, test_pred))

if args.total_performance:
    save_name = f"{BASE_PATH}/" \
                f"General_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/" \
                f"graphsage_metrics_emb{dis_embs_values.shape[1]}.pickle"
    with open(save_name, 'wb') as handle:
        pickle.dump(metrics_general, handle, protocol=pickle.HIGHEST_PROTOCOL)

    save_name = f"{BASE_PATH}/" \
                f"CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/" \
                f"graphsage_metrics_emb{dis_embs_values.shape[1]}.pickle"
    with open(save_name, 'wb') as handle:
        pickle.dump(metrics_dependencies, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif args.general_performance:
    save_name = f"{BASE_PATH}/" \
                f"General_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/" \
                f"graphsage_metrics_emb{dis_embs_values.shape[1]}.pickle"
    with open(save_name, 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    save_name = f"{BASE_PATH}/"\
                f"CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ','_')}/"\
                f"graphsage_metrics_emb{dis_embs_values.shape[1]}.pickle"

    with open(save_name, 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
