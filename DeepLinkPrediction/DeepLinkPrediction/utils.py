from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.split import EvalSplit
from sklearn.externals.joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from scipy.stats import hypergeom, wilcoxon, kruskal
from functools import reduce
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix
from DeepLinkPrediction.DLembedder import DLembedder, TimingCallback
from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
from itertools import product, permutations
from random import sample
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gseapy as gp
import collections
import keras
import random
import h5py
import json
import glob
import networkx as nx
import ndex2
import os


def read_dependencies(dependencies_fp, ppi_scaffold_fp, rnai=True):
    """
    Read in gene dependency scores from CRISPR or RNAi datasets
    :param dependencies_fp:
    :param ppi_scaffold_fp:
    :param rnai:
    :return: dependency df and UndirectedInteractionNetwork object
    """
    if rnai:
        dep_df = pd.read_csv(dependencies_fp, header=0, sep=',',
                                index_col=0).transpose().dropna(how='all')
    else:
        dep_df = pd.read_csv(dependencies_fp, header=0, index_col=0).dropna(how='all')
    # dep_df.fillna(0, inplace=True)
    dep_df.columns = [i.split(' ')[0] for i in dep_df.columns]
    if type(ppi_scaffold_fp) == str:
        nw_df_hgnc = pd.read_csv(ppi_scaffold_fp, sep='\t', header=0, index_col=0)
        nw_obj = UndirectedInteractionNetwork(nw_df_hgnc)
    else:
        nw_obj = ppi_scaffold_fp
    common_genes = set(dep_df.columns) & set(nw_obj.node_names)
    dep_df = dep_df[common_genes]

    if type(ppi_scaffold_fp) == str:
        return dep_df, nw_obj
    else:
        return dep_df


def read_ppi_scaffold(ppi_name, ppi_fp):
    if ppi_name == "STRING":
        ndex_nw_df = pd.read_csv(ppi_fp + ppi_name + '.tsv', sep='\t', header=0, index_col=0)
    elif ppi_name == "reactome":
        ndex_nw_df = pd.read_csv(ppi_fp + ppi_name + '.txt', sep='\t', header=0, usecols=['Gene1', 'Gene2'])
    elif ppi_name == "bioplex":
        ndex_nw_df = pd.read_csv(ppi_fp + ppi_name + '.tsv', sep='\t', header=0, usecols=['SymbolA', 'SymbolB'])
    else:
        temp = ndex2.create_nice_cx_from_file(ppi_fp + ppi_name + '.cx')
        ndex_nw_df = temp.to_pandas_dataframe().drop("interaction", axis=1)
        del temp

    G = nx.Graph()
    G.add_edges_from(ndex_nw_df.values)

    Gcc = G.subgraph(max(nx.connected_components(G), key=len))
    return UndirectedInteractionNetwork(pd.DataFrame(Gcc.edges, columns=['GeneA', 'GeneB']))


def extract_pos_dict_at_threshold(dep_df, threshold=-1.5):
    """
    :param dep_df:
    :param threshold:
    :return: dictionary where key = cell line and value = dependencies below certain threshold
    """
    pos = {}
    min_no_deps = 3
    for cl in dep_df.index:
        tmp_pos = dep_df.loc[cl][dep_df.loc[cl] < threshold].index.tolist()
        if tmp_pos and len(tmp_pos) > min_no_deps - 1:
            pos[cl] = tmp_pos
        else:
            print(f"For cell line {cl} {len(tmp_pos)} postives are found at threshold {threshold}")
            continue
    return pos


def extract_interm_dict_at_threshold(dep_df, pos_dict, pos_threshold=-1.5, neg_threshold=-0.5):
    """

    :param dep_df:
    :param threshold:
    :return: dictionary where key = cell line and value = dependencies below certain threshold
    """
    dep_df = dep_df.loc[pos_dict.keys()]
    intermediate = {}
    for cl in dep_df.index:
        nans = set(dep_df.loc[cl][dep_df.loc[cl].isna()].index)
        tmp_interm = dep_df.loc[cl][
            (dep_df.loc[cl] < neg_threshold) & (dep_df.loc[cl] > pos_threshold)].index.tolist()
        intermediate[cl] = list(set(tmp_interm) - nans)
    return intermediate


def extract_negatives_dict_at_threshold(dep_df, pos_dict, neg_threshold=-0.5):
    dep_df = dep_df.loc[pos_dict.keys()]
    negs = {}
    for cl in dep_df.index:
        nans = set(dep_df.loc[cl][dep_df.loc[cl].isna()].index)
        tmp_negs = dep_df.loc[cl][(dep_df.loc[cl] > neg_threshold)].index.tolist()
        negs[cl] = list(set(tmp_negs) - nans)
    return negs


def updated_gene2int(gene2int, dep_df):
    """
    Updates the gene2int that is necessary for the deep link prediction model to known which genes AND
    cell lines correspond to which index
    :param gene2int: `old` gene2int from the nw object
    :param dep_df: dependency dataframe with columns = cell lines
    :return: updated gene2int and int2gene
    """
    cellLine_index = np.arange(max(gene2int.values()) + 1, max(gene2int.values()) + dep_df.shape[0] + 1)
    gene2int.update({i: v for i, v in zip(dep_df.index, cellLine_index)})
    int2gene = {v: k for k, v in gene2int.items()}
    return gene2int, int2gene


def train_model(undir_obj,  seed, npr=None, train_ratio=0.8, train_data=None,
                validation_data=None, int2gene=None, N_nodes=None, merge_method='ABS-DIFF'):
    """
    Function to train the model using DLembedder
    :param undir_obj: UnderectedInteractionNetwork object
    :param seed: random seed to use for reproducibility
    :param npr: negative:positie ratio
    :param train_ratio: training ratio
    :param train_data: Optional, if given train on this data else make the split in this function
    :param validation_data: Optional, if given validate model on this dataset. NOTE: this is NOT the test dataset,
    the test dataset is calculated by the model itself!
    :param int2gene: Optional, dictionary that maps index to gene name
    :param N_nodes: Optional, number of nodes in the network
    :param merge_method: Optional, which merge method to use to construct the edge embeddings
    :return:
    """

    if int2gene is None:
        int2gene = undir_obj.int2gene
    if train_data is None:
        X_train, X_test, Y_train, Y_test, _ = undir_obj.getTrainTestData(neg_pos_ratio=npr,
                                                                         method='ms_tree',
                                                                         train_ratio=train_ratio)
    else:
        X_train, Y_train = train_data[0], train_data[1]

    if N_nodes:
        N_nodes = N_nodes
    else:
        N_nodes = undir_obj.N_nodes

    dl_net = DLembedder(N_nodes, 10, nodes_per_layer=[32, 32, 1],
                        activations_per_layer=['relu', 'relu', 'sigmoid'], int2genedict=int2gene,
                        dropout=0.2, merge_method=merge_method, random_state=seed)

    dl_net.counter = 0

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
    time_cb = TimingCallback()

    if validation_data:
        _ = dl_net.fit(X_train, Y_train, validation_data=validation_data, callbacks=[earlyStopping, time_cb],
                            n_epochs=50, verbose=2, allow_nans=False, metrics='binary_accuracy',
                            loss='binary_crossentropy', custom_loss=False)
    else:
        _ = dl_net.fit(X_train, Y_train, validation_split=0.2, callbacks=[earlyStopping, time_cb],
                                n_epochs=50, verbose=2, allow_nans=False, metrics='binary_accuracy',
                                loss='binary_crossentropy', custom_loss=False)

    if train_data:
        return dl_net
    else:
        return dl_net, X_test, Y_test


def generate_traintest_dependencies(dependency_data, threshold_neg, threshold_pos, npr, gene2int,
                                    train_test_ratio=0.8, train_validaiton_ratio=None, exclude_negs=None):
    """
    Returns a number of negative and positive interactions based on dependency threshold and a predefined negative:positive ratio

    :param dependency_data: dataframe with cols=genes, index=cell line
    :param pos_dict: dictionary containing positives per cell line
    :param threshold: dependency log2 threshold
    :param npr: negative to positive ratio for dependencies
    :param gene2int: dictionary that maps each node to its index
    :return: array of negative interactions
    """
    min_no_deps = 3
    pos = {}
    negs = {}
    intermediate = {}
    for cl in dependency_data.index:
        nans = set(dependency_data.loc[cl][dependency_data.loc[cl].isna()].index)
        tmp_pos = dependency_data.loc[cl][dependency_data.loc[cl] < threshold_pos].index.tolist()
        if tmp_pos and len(tmp_pos) > min_no_deps-1:
            pos[cl] = tmp_pos
        else:
            # print(
            #     f"For cell line {cl} {len(tmp_pos)} postives are found at threshold {threshold_pos}, increasing threshold by 0.5")
            # thresh_pos_new = threshold_pos
            # while len(tmp_pos) < min_no_deps:
            #     thresh_pos_new += + 0.5
            #     tmp_pos = dependency_data.loc[cl][dependency_data.loc[cl] < thresh_pos_new].index.tolist()
            # print(f"For cell line {cl}, {len(tmp_pos)} positives were found at threshold {thresh_pos_new}")
            # pos[cl] = tmp_pos
            print(f"For cell line {cl} {len(tmp_pos)} postives are found at threshold {threshold_pos}")
            continue
        N_negs = np.int(npr * len(pos[cl]))
        all_negs = dependency_data.loc[cl][dependency_data.loc[cl] > threshold_neg].index.tolist()
        if len(all_negs) < N_negs:
            print(f"Too few negatives available, taking the maximum possible: {len(all_negs)}")
            negs[cl] = list(set(all_negs) - nans)
        else:
            negs[cl] = list(set(random.sample(all_negs, N_negs)) - nans)

        tmp_interm = dependency_data.loc[cl][
            (dependency_data.loc[cl] < threshold_neg) & (dependency_data.loc[cl] > threshold_pos)].index.tolist()
        intermediate[cl] = list(set(tmp_interm) - nans)
        # pdb.set_trace()

    if exclude_negs:
        negs = {k: list(set(sample(v, len(v)))-exclude_negs) for k, v in negs.items()}
    pos = {k:sample(v, len(v)) for k, v in pos.items()}
    intermediate = {k: sample(v, len(v)) for k, v in intermediate.items()}


    neg_thresh = calculate_traintestval_thresholds(negs, train_test_ratio, train_validaiton_ratio) # cl = tuple((ttr, tvr))
    pos_thresh = calculate_traintestval_thresholds(pos, train_test_ratio, train_validaiton_ratio)
    interm_thresh = calculate_traintestval_thresholds(intermediate, train_test_ratio, train_validaiton_ratio)

    if train_test_ratio is not None:
        negs_arr_train = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in negs.items() for gene in
             genes[:neg_thresh[celline][0]]])
        pos_arr_train = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in pos.items() for gene in
                                  genes[:pos_thresh[celline][0]]])
        intermediate_arr_train = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in intermediate.items() for gene in
             genes[:interm_thresh[celline][0]]])

        negs_arr_test = np.array([[gene2int[celline], gene2int[gene]] for celline, genes in negs.items() for gene in
                                 genes[neg_thresh[celline][0]:]])
        pos_arr_test = np.array([[gene2int[celline], gene2int[gene]] for celline, genes in pos.items() for gene in
                                genes[pos_thresh[celline][0]:]])
        intermediate_arr_test = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in intermediate.items() for gene in
             genes[interm_thresh[celline][0]:]])

        if train_validaiton_ratio is not None:
            negs_arr_train = np.array(
                [[gene2int[celline], gene2int[gene]] for celline, genes in negs.items() for gene in
                 genes[:neg_thresh[celline][1]]])
            pos_arr_train = np.array(
                [[gene2int[celline], gene2int[gene]] for celline, genes in pos.items() for gene in
                 genes[:pos_thresh[celline][1]]])
            intermediate_arr_train = np.array(
                [[gene2int[celline], gene2int[gene]] for celline, genes in intermediate.items() for gene in
                 genes[:interm_thresh[celline][1]]])

            negs_arr_val = np.array(
                [[gene2int[celline], gene2int[gene]] for celline, genes in negs.items() for gene in
                 genes[neg_thresh[celline][1]:neg_thresh[celline][0]]])
            pos_arr_val = np.array(
                [[gene2int[celline], gene2int[gene]] for celline, genes in pos.items() for gene in
                 genes[pos_thresh[celline][1]:pos_thresh[celline][0]]])
            intermediate_arr_val = np.array(
                [[gene2int[celline], gene2int[gene]] for celline, genes in intermediate.items() for gene in
                 genes[interm_thresh[celline][1]:interm_thresh[celline][0]]])

            return negs, negs_arr_train, negs_arr_val, negs_arr_test, pos, pos_arr_train, pos_arr_val, pos_arr_test,\
                       intermediate, intermediate_arr_train, intermediate_arr_val, intermediate_arr_test

        return negs, negs_arr_train, negs_arr_test, pos, pos_arr_train, pos_arr_test, \
                   intermediate, intermediate_arr_train, intermediate_arr_test

    else:
        negs_arr = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in negs.items() for gene in genes])
        pos_arr = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in pos.items() for gene in genes])
        intermediate_arr = np.array(
            [[gene2int[celline], gene2int[gene]] for celline, genes in intermediate.items() for gene in genes])

        return negs, negs_arr, pos, pos_arr, intermediate, intermediate_arr


def construct_combined_traintest(pos_arr_train, negs_arr_train, X_train_, Y_train_,
                                 pos_arr_val=None, negs_arr_val=None, X_val_=None, Y_val_=None,
                                 pos_arr_test=None, negs_arr_test=None, X_test_=None, Y_test_=None):
    """
    Construct combined PPI & dependencies train/test test
    :param pos_arr_train: Positive TRAIN dependencies as returned by function generate_traintest_dependencies
    :param pos_arr_val: Positive TEST dependencies as returned by function generate_traintest_dependencies
    :param negs_arr_train: Negative TRAIN dependencies as returned by function generate_traintest_dependencies
    :param negs_arr_val: Negative TEST dependencies as returned by function generate_traintest_dependencies
    :param X_train_: train PPI interactions
    :param Y_train_: train PPI labels
    :param X_test_: test PPI interactions
    :param Y_test_: test PPI labels
    :return: Final X_train, y_train, X_test, y_test
    """
    assert len(
        set(map(tuple, pos_arr_train)) & set(map(tuple, negs_arr_train))) == 0, "Error overlapping neg pos TRAIN interactions"
    if pos_arr_val is not None and negs_arr_val is not None:
        assert len(
            set(map(tuple, pos_arr_val)) & set(map(tuple, negs_arr_val))) == 0, "Error overlapping neg pos VAL interactions"
    if pos_arr_test is not None and negs_arr_test is not None:
        assert len(
            set(map(tuple, pos_arr_test)) & set(map(tuple, negs_arr_test))) == 0, "Error overlapping neg pos test interactions"

    train_dependencies = np.vstack((pos_arr_train, negs_arr_train))
    train_labels = np.hstack(
        (np.ones(pos_arr_train.shape[0]), np.zeros(negs_arr_train.shape[0])))
    assert train_dependencies.shape[0] == train_labels.shape[0], 'ERROR'

    if pos_arr_val is not None and negs_arr_val is not None and pos_arr_val.shape[0] != 0:
        val_dependencies = np.vstack((pos_arr_val, negs_arr_val))
        val_labels = np.hstack(
            (np.ones(pos_arr_val.shape[0]), np.zeros(negs_arr_val.shape[0])))
        assert val_dependencies.shape[0] == val_labels.shape[0], 'ERROR'
    else:
        val_dependencies = None
        val_labels = None

    if pos_arr_test is not None and negs_arr_test is not None and pos_arr_test.shape[0] != 0:
        test_dependencies = np.vstack((pos_arr_test, negs_arr_test))
        test_labels = np.hstack(
            (np.ones(pos_arr_test.shape[0]), np.zeros(negs_arr_test.shape[0])))
        assert test_dependencies.shape[0] == test_labels.shape[0], 'ERROR'
    else:
        test_dependencies = None
        test_labels = None

    X_train = np.vstack((X_train_, train_dependencies))
    if X_test_ is not None and X_test_.shape[0] != 0:
        X_test = np.vstack((X_test_, test_dependencies))
    else:
        X_test = None
    if X_val_ is not None and X_val_.shape[0] != 0:
        X_val = np.vstack((X_val_, val_dependencies))
    else:
        X_val = None

    y_train = np.hstack((Y_train_, train_labels))
    if Y_test_ is not None and Y_test_.shape[0] != 0:
        y_test = np.hstack((Y_test_, test_labels))
    else:
        y_test = None
    if Y_val_ is not None and Y_val_.shape[0] != 0:
        y_val = np.hstack((Y_val_, val_labels))
    else:
        y_val = None

    if X_test_ is not None:
        if X_val_ is not None:
            return X_train, X_val, X_test, y_train, y_val, y_test
        return X_train, X_test, y_train, y_test
    else:
        return X_train, y_train


def pretty_print(ap, auc, acc, f1):
    print('\n' + '#' * 9 + ' Link Prediction Performance ' + '#' * 9)
    print(f'AUC-ROC: {auc:.3f}, AUC-PR: {ap:.3f}, Accuracy: {acc:.3f}, F1-score: {f1:.3f}')
    print('#' * 50)


def getModelPerformance(Y_true, preds):
    if type(preds) == list:
        preds = np.array(preds)
    average_pr = average_precision_score(Y_true, preds)
    auc = roc_auc_score(Y_true, preds)
    acc = accuracy_score(Y_true, (preds > 0.5).astype(np.int_))
    cm = confusion_matrix(Y_true, (preds > 0.5).astype(np.int_))
    f1 = f1_score(Y_true, (preds > 0.5).astype(np.int_))
    return average_pr, auc, acc, cm, f1


def predict_dependencies(negatives, positives, dl_net, gene2int, cne=False, verbose=True):
    """
    Predicts the intermediate (label 0) and postive (label 1) dependencies
    :param negatives: DICT of key = cell line, value = negatives dependencies
    :param positives: DICT of key = cell line, value = positive dependencies
    :param dl_net: DLembedder object
    :param gene2int: updated gene2int as returned by function updated_gene2int
    :return: model performance
    """
    model_performance = {}
    for cl, neg in negatives.items():
        to_pred_genes = neg + positives[cl]
        perc_pos = len(positives[cl]) / len(to_pred_genes)
        labels = np.array([0] * len(neg) + [1] * len(positives[cl]))
        to_pred = np.array(list(product([gene2int[cl]], [gene2int[i] for i in to_pred_genes])))
        if cne:
            preds = dl_net.predict(to_pred)
            ap_, auc_, acc_, cm_, f1_ = getModelPerformance(Y_true=labels, preds=preds)
        else:
            ap_, auc_, acc_, cm_, f1_ = dl_net.getModelPerformance(Y_true=labels, preds=preds, metric='all')
            preds = dl_net.predict_proba(to_pred)
        if verbose:
            pretty_print(ap_, auc_, acc_, f1_)
        model_performance[cl] = tuple((ap_, auc_, acc_, cm_, f1_, perc_pos * 100, to_pred_genes, preds))
    return model_performance


def read_h5py(fp, dtype=int):
    hf = h5py.File(fp, 'r')
    x = np.array(hf.get(fp.split('/')[-1][:-5]), dtype=dtype)
    hf.close()
    return x


def write_h5py(fp, data):
    hf = h5py.File(fp, 'w')
    hf.create_dataset(fp.split('/')[-1][:-5], data=data)
    hf.close()
    return None


def get_targets(ot, disease, thresholds, datatypes, depmap_id2ccle_name, dis_df, verbose=True, take_overlap=True):
    d = ot.get_associations_for_disease(disease)
    bc_targets_ot = []
    for i in d:
        if len(datatypes) == 1:
            if i['association_score']['datatypes'][datatypes[0]] >= thresholds[0]:
                # if i['association_score']['datatypes']['known_drug'] >= 0.8 and i['association_score']['datatypes']['somatic_mutation'] >= 0.8:
                bc_targets_ot.append(i['target']['gene_info']['symbol'])

    if verbose:
        print(f"{len(bc_targets_ot)} number of targets retrieved for {disease}")

    with open('../../depmap_data/target_per_cell_line.txt', 'r') as f:
        all_targets = json.load(f)
    all_targets_ccle = {depmap_id2ccle_name[k]: v for k, v in all_targets.items() if
                        k in depmap_id2ccle_name}
    if verbose:
        print(f"{len(all_targets)-len(all_targets_ccle)} cell lines lost in ID mapping")

    if take_overlap:
        json_targets = {k: set(bc_targets_ot) & set(v) for k, v in all_targets_ccle.items() if k in dis_df.index}
    else:
        json_targets = {k: set(v) for k, v in all_targets_ccle.items() if k in dis_df.index}

    if verbose:
        print(f"{len(all_targets_ccle)-len(json_targets)} cell lines not in {disease} cell lines")
        print(f"We are left with {len(json_targets)} cell lines")
    return json_targets


def calculate_traintestval_thresholds(dep_dict, train_test_ratio, train_validation_ratio):
    out_d = {}

    if train_test_ratio is not None:
        for cl, genes in dep_dict.items():
            ttr = int(len(genes) * train_test_ratio)
            if train_validation_ratio is not None:
                tvr = int(ttr * train_validation_ratio)
            else:
                tvr = None
            out_d[cl] = tuple((ttr, tvr))
    else:
        return None

    return out_d


def construct_cellline_splits_all(dep_dict, pos_dep_dict, gene2int, deps='negpos', fp=None, return_d=False):
    out_d = {}
    tmp = []
    for cl, neg in dep_dict.items():
        to_pred_genes = neg + pos_dep_dict[cl]
        assert not set(neg) & set(pos_dep_dict[cl])
        labels = np.array([0] * len(neg) + [1] * len(pos_dep_dict[cl]))
        to_pred = np.array(list(product([gene2int[cl]], [gene2int[i] for i in to_pred_genes])))
        tmp.append(to_pred)
        assert pd.DataFrame(to_pred).shape == pd.DataFrame(to_pred).drop_duplicates().shape
        out_d[cl] = tuple((to_pred_genes, to_pred, labels))
        if fp is not None:
            write_h5py(fp+f'{cl}_{deps}.hdf5', data=to_pred, cellline=True)
            write_h5py(fp+f'{cl}_{deps}_labels.hdf5', data=labels, cellline=True)
    tmp_ = np.vstack(tmp)
    if return_d:
        return out_d
    else:
        return None


def metric_df_across_diseases(metric_d, metric_title, fp):
    auroc_df = pd.concat(metric_d.values(), axis=1)
    auroc_df.columns = [i.replace('_', ' ') for i in metric_d.keys()]
    auroc_df['Mean '+metric_title] = auroc_df.mean(axis=1).values
    auroc_df.sort_values('Mean '+metric_title, ascending=False).to_csv(fp)
    return None


def construct_drug_shared_heatmaps(drug_d1, methods, drug_d2=None, savepath=None, title=None):
    drug_df1 = pd.DataFrame(np.eye(len(methods)), columns=methods, index=methods)
    numbered_drug_df1 = pd.DataFrame.from_dict({k: len(v) for k, v in drug_d1.items()}, orient='index').loc[
        drug_df1.index]
    annot = np.zeros((len(methods), len(methods))).astype(str)
    annot[annot == '0.0'] = ''
    annot[np.diag_indices_from(annot)] = numbered_drug_df1.values.ravel().astype(str)

    for a, b in permutations(methods, 2):
        if drug_d1[b]:
            drug_df1.loc[a, b] = len(set(drug_d1[a]) & set(drug_d1[b])) / len(drug_d1[b])
        else:
            drug_df1.loc[a, b] = 0

    if drug_d2 is not None:
        drug_df2 = pd.DataFrame(np.eye(len(methods)), columns=methods, index=methods)
        for a, b in permutations(methods, 2):
            if drug_d2[b]:
                drug_df2.loc[a, b] = len(set(drug_d2[a]) & set(drug_d2[b])) / len(drug_d2[b])
            else:
                drug_df2.loc[a, b] = 0

        fig, axn = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 8))
        sns.heatmap(drug_df1, ax=axn[0], annot=False)
        sns.heatmap(drug_df2, ax=axn[1], annot=False)

    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(drug_df1, ax=ax, annot=annot, cbar=True, robust=True, center=0, fmt='')

    plt.title(title)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


def plot_twoway_barplot(pval_df, pval_df_ori, sorted_ii, savepath=None):
    y = np.arange(pval_df.shape[0])
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=((10, 5)))
    axes[0].barh(y, pval_df_ori.loc[sorted_ii, 'significant_count'], align='center', color='gray', zorder=10)
    axes[0].set(title='ORIGINAL>METHOD')
    axes[1].barh(y, pval_df.loc[sorted_ii, 'significant_count'], align='center', color='gray', zorder=10)
    axes[1].set(title='METHOD>ORIGINAL')

    axes[0].invert_xaxis()
    axes[0].set(yticks=y, yticklabels=sorted_ii)
    axes[0].yaxis.tick_right()

    for ax in axes.flat:
        ax.margins(0.03)
        ax.grid(True)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


def get_topK_intermediaries(df, cl, topk, t, original=True, top=True, cl_list=None):
    if top:
        if not original:
            df_ = df.sort_values('Mean', ascending=False)
            df_ = df_.reset_index(drop=True)
            grouped_by_cl = df_.groupby('TestEdges_A').groups
            for k, row in df_.iloc[grouped_by_cl[cl]].iterrows():
                if row.TestEdges_B not in cl_list:
                    # print(row.TestEdges_B)
                    topk.add(row.TestEdges_B)
                    if len(topk) == t:
                        return topk
        else:
            for k, row in df.loc[cl].sort_values(ascending=True).iteritems():
                if k not in cl_list:
                    topk.add(k)
                    if len(topk) == t:
                        return topk
    else:
        if not original:
            df.sort_values('Mean', ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
            grouped_by_cl = df.groupby('TestEdges_A').groups
            for k, row in df.iloc[grouped_by_cl[cl]].iterrows():
                if row.TestEdges_B not in cl_list:
                    topk.add(row.TestEdges_B)
                    if len(topk) == t:
                        return topk
        else:
            for k, row in df.loc[cl].sort_values(ascending=False).iteritems():
                if k not in cl_list:
                    topk.add(k)
                    if len(topk) == t:
                        return topk


def plot_pmf(x, y_1, y1_label, y_2=None, y2_label=None, title=None, save_fp=None, save_raw_data=None):

    if save_raw_data:
        pd.DataFrame({y1_label:y_1, y2_label:y_2}).to_csv(save_raw_data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y_1, 'bo', label=y1_label)
    ax.vlines(x, 0, y_1, lw=1)
    if y_2 is not None:
        ax.plot(x, y_2, 'ro', label=y2_label)
        ax.vlines(x, 0, y_2, lw=1)

    ax.set_xlabel('# of targets in our group of chosen genes')
    ax.set_ylabel('hypergeom PMF')
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if save_fp is None:
        plt.show()
    else:
        plt.savefig(save_fp, dpi=300)
        plt.close(fig)


def run_permutations(N_PERMUTATIONS, options, pick_size, sens_targets, methods, total_sensitive_targets_retrieved,
                     cl, disease, weights=None, seed=23):
    np.random.seed(seed)
    tmp_withdegree = []  # vector of len = 10k
    tmp_uniform = []
    for _ in range(N_PERMUTATIONS):
        topK_random_withdegree = set(np.random.choice(options, size=pick_size, p=weights, replace=False))
        topK_random_uniform = set(np.random.choice(options, size=pick_size, p=None, replace=False))
        overlap_withdegree = len(topK_random_withdegree & sens_targets)
        overlap_uniform = len(topK_random_uniform & sens_targets)
        tmp_withdegree.append(overlap_withdegree)
        tmp_uniform.append(overlap_uniform)

    uniks_withdegree, counts_withdegree = np.unique(tmp_withdegree, return_counts=True) # vector in which each element represents targets retrieved using random top 100
    weighted_average_with_degree = np.average(uniks_withdegree, weights=counts_withdegree)
    uniks_uniform, counts_uniform = np.unique(tmp_uniform, return_counts=True)

    if uniks_withdegree.shape[0] > uniks_uniform.shape[0]:
        x = uniks_withdegree
        counts_uniform = np.array(list(counts_uniform) +
                                  [0] * (uniks_withdegree.shape[0] - uniks_uniform.shape[0]))
    else:
        x = uniks_uniform
        counts_withdegree = np.array(list(counts_withdegree) +
                                     [0] * (uniks_uniform.shape[0] - uniks_withdegree.shape[0]))

    plot_pmf(x, counts_withdegree / N_PERMUTATIONS, 'degree-based', counts_uniform / N_PERMUTATIONS, 'uniform',
             title=None,
             save_fp=f"drug_sensitivity_data/pmfs/lung_cancer_{cl}",
             save_raw_data=f"drug_sensitivity_data/pmfs/lung_cancer_{cl}_raw.csv")

    random_significantly_better = {}
    random_significantly_worse = {}
    better_than_random_setting = {}
    for method in methods:
        no_targets_found = len(total_sensitive_targets_retrieved[disease][cl][method])
        if no_targets_found > np.max(uniks_withdegree):
            random_significantly_better[method] = 0
            random_significantly_worse[method] = 1
        else:
            random_significantly_better[method] = \
                counts_withdegree[np.where(uniks_withdegree >= no_targets_found)[0][0]:].sum() / N_PERMUTATIONS
            random_significantly_worse[method] = \
                counts_withdegree[:np.where(uniks_withdegree <= no_targets_found)[0][0]+1].sum() / N_PERMUTATIONS

        if no_targets_found > weighted_average_with_degree:
            better_than_random_setting[method] = no_targets_found - weighted_average_with_degree
        else:
            better_than_random_setting[method] = 0

    return random_significantly_better, random_significantly_worse, better_than_random_setting


def construct_mean_predictions_df(save_preds_fp, method, heterogeneous_network_obj):
    method_fp = [f for f in glob.glob(save_preds_fp) if method in f][0]
    f_l = sorted(glob.glob(f'{method_fp}/{method}*_test_genes*'))
    f_p = sorted(glob.glob(f'{method_fp}/{method}*_test_preds*'))
    all_df = []
    for repeat in range(len(f_l)):
        # repeat = 1
        test_edges = np.loadtxt(f_l[repeat], delimiter=',', dtype=int)
        test_preds = np.loadtxt(f_p[repeat], delimiter=',', dtype=float)

        tot_df = pd.DataFrame({'TestEdges_A': test_edges[:, 0], 'TestEdges_B': test_edges[:, 1]})
        tot_df = tot_df.applymap(lambda x: heterogeneous_network_obj.int2gene[int(x)])
        tot_df['Predictions'] = test_preds
        # tot_df['Labels'] = final_labels
        all_df.append(tot_df)
    all_df_v2 = reduce(lambda left, right: pd.merge(left, right,
                                                    on=['TestEdges_A', 'TestEdges_B']), all_df)
    all_df_v2['Mean'] = all_df_v2[[i for i in list(all_df_v2) if i.startswith('Predictions')]].mean(axis=1)
    return all_df_v2


def calculate_stacked_barplot(rand_d_better, rand_d_worse, common_cls, disease,
                         methods_nice_name_d=None, save_raw_data=None):

    times_significant_better = pd.DataFrame(rand_d_better[disease]).applymap(
        lambda x: x < 0.05).sum(axis=1).sort_values()
    times_significant_worse = pd.DataFrame(rand_d_worse[disease]).applymap(
        lambda x: x < 0.05).sum(axis=1).sort_values()
    times_significant = pd.concat([times_significant_better, times_significant_worse], axis=1, join='inner')
    times_significant.columns = ["significantly better", "significantly worse"]
    times_significant['neutral'] = len(common_cls) - times_significant["significantly better"] - \
                                   times_significant["significantly worse"]

    times_significant = times_significant[["significantly worse", "neutral", "significantly better"]]
    times_significant["sort_on_this"] = times_significant[
        "significantly better"]  # + times_significant['better_average']
    times_significant.sort_values(by='sort_on_this', inplace=True)
    times_significant.drop('sort_on_this', axis=1, inplace=True)
    times_significant.index = [method if method not in methods_nice_name_d else methods_nice_name_d[method] for method
                               in
                               times_significant.index]

    if save_raw_data is not None:
        times_significant.to_csv(save_raw_data)

    return times_significant


def plot_stacked_barplot(times_significant, save_fp=None, ax=None, annotation=None, pdf=True, legend=True):
    # clrs = random.sample(sns.color_palette('colorblind'), 3)
    clrs = [(0.792156862745098, 0.5686274509803921, 0.3803921568627451),
            (0.00784313725490196, 0.6196078431372549, 0.45098039215686275)]

    if ax is None:
        pa = times_significant.plot.barh(stacked=True, figsize=(8.5, 3),
                                         color=clrs)
    else:
        pa = times_significant.plot.barh(stacked=True, figsize=(8.5, 3),
                                    color=clrs, ax=ax, legend=legend)
    bars = ax.patches
    print(bars)
    patterns = ['/', '\\']
    hatches = []
    for h in patterns:  # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    for bar, hatch in zip(bars, hatches):  # loop over bars and hatches to set hatches in correct order
        bar.set_hatch(hatch)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)

    if annotation is not None:
        ax.annotate(annotation, xy=(-0.2, 1), xycoords='axes fraction', fontsize=8,
                    horizontalalignment='left', verticalalignment='top')

    ax.set_xlabel("Number of cell lines", fontsize=6)
    plt.legend(bbox_to_anchor=(1.05, 1), prop={'size': 6})
    # plt.title(title)
    plt.tight_layout()

    if ax is None:
        if save_fp is not None:
            save_fp = save_fp + ".pdf" if pdf else save_fp
            plt.savefig(save_fp, dpi=300)
            plt.close()
        else:
            plt.show()


def plot_annotated_barplot(sens_d, disease, title=None, non_sens_d=None, save_fp=None, annot_text=None,
                           methods_significant=None):
    methods_nice_name_d = {'line-opene': 'LINE', 'n2v-opene': 'node2vec', 'deepwalk-opene': 'DeepWalk',
                           'grarep-opene': 'GraRep'}
    if isinstance(sens_d, dict):
        sens_targ = pd.DataFrame(sens_d[disease])
        sens_targ.index = [method if method not in methods_nice_name_d else methods_nice_name_d[method] for method in
                           sens_targ.index]
        sens_targ = sens_targ.applymap(lambda x: len(x))

        if methods_significant is None:
            pval_ori_df = calculate_significance_vs_original(sens_targ)
            methods_significant = pval_ori_df[pval_ori_df['MWU P-value'] < 0.05].Method.values

        if non_sens_d is not None:
            nonsens_targ = pd.DataFrame.from_dict(non_sens_d[disease])
            nonsens_targ = nonsens_targ.applymap(lambda x: len(x))

            targ_retriev = pd.concat([sens_targ.mean(axis=1).to_frame("Sensitive"),
                                      nonsens_targ.mean(axis=1).to_frame("NON-Sensitive")],
                                     axis=1).sort_values("Sensitive", ascending=True)
            assert np.array_equal(sens_targ.index, nonsens_targ.index)
            targ_retriev['method'] = targ_retriev.index
            targ_retriev = targ_retriev.round(decimals=0)
            targ_retriev['ratios'] = targ_retriev['Sensitive'].map(int).map(str) + ':' + \
                                     targ_retriev['NON-Sensitive'].map(int).map(str)
            heights = (targ_retriev['Sensitive'] + targ_retriev['NON-Sensitive']).values
        else:
            targ_retriev = sens_targ.mean(axis=1).to_frame("Sensitive").sort_values("Sensitive", ascending=True)
            heights = targ_retriev['Sensitive']
    else:
        targ_retriev = sens_d
        heights = targ_retriev.values

    ax = targ_retriev.plot.barh(stacked=True, figsize=(6,4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if annot_text is not None:
        for p, val, height, method in zip(ax.patches, annot_text.loc[targ_retriev.index, 'annot_text'], heights, targ_retriev.index):
            if method in methods_significant:
                ax.annotate(val, (height + 1.3, p.get_y() + 0.25), ha='center', va='center', fontweight='bold')
            else:
                ax.annotate(val, (height + 1.3, p.get_y() + 0.25), ha='center', va='center')
    else:
        if non_sens_d is not None:
            for p, val, height in zip(ax.patches, targ_retriev['ratios'], heights):
                ax.annotate(val, (height + 0.5, p.get_y() + 0.25), ha='center', va='center')
        else:
            for p, val, height in zip(ax.patches, heights, heights):
                ax.annotate(val, (height + 0.5, p.get_y() + 0.25), ha='center', va='center')

    plt.xlim(0, max(heights) + 5)
    plt.tight_layout()
    plt.title(title)
    plt.legend(loc='lower right')
    if save_fp is not None:
        plt.savefig(save_fp, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_significance_vs_original(df, original_method='original', fdr=0.05):
    pval_vs_original = []
    for m in df.index:
        v = df.loc[m]
        if m != original_method:
            if not np.array_equal(v, df.loc[original_method]):
                # _, pval = mannwhitneyu(v, df.loc['original'], alternative='greater')
                _, pval = wilcoxon(x=v, y=df.loc[original_method], alternative='greater')
                # print(m, pval, len(v))
            else:
                pval = np.inf
            pval_vs_original.append(
                pd.DataFrame([[m, pval, (v > 0).sum()]], columns=['Method', 'MWU P-value', 'no. Cell lines']))
        else:
            pval_vs_original.append(
                pd.DataFrame([[m, np.inf, (v > 0).sum()]], columns=['Method', 'MWU P-value', 'no. Cell lines']))

    if fdr is not None:
        df = pd.concat(pval_vs_original)
        df.index = df.Method
        _, qvals, _, _ = multipletests(df.drop(original_method)["MWU P-value"].values, fdr, method='fdr_bh')
        df['FDR'] = np.insert(qvals, 0, 1)
        return df
    else:
        return pd.concat(pval_vs_original)


def get_mean_performance_df(perf_obj, metrics):
    dfs_l = []
    for metric in metrics:
        t = perf_obj.get_pandas_df(metric=metric)
        t.columns = [metric]
        dfs_l.append(t)

    mean_df = pd.concat(dfs_l, axis=1)
    mean_df.columns = [i.capitalize() for i in mean_df.columns]
    mean_df.iloc[:, 4:12] = mean_df.iloc[:, 4:12] * 100
    mean_df = mean_df.astype(dtype={'Tn': "int64", 'Fp': "int64", 'Fn': "int64", 'Tp': "int64", 'Auroc': "float64",
                                    'Precision': "float64", 'Recall': "float64", 'Fallout': "float64",
                                    'Miss': "float64",
                                    'Accuracy': "float64", 'F_score': "float64", 'Average_precision': "float64",
                                    'Eval_time': "int64"})
    return mean_df


# def get_run_performance(perf_obj, metric):


def plot_heatmap_performance_values(perf_d_1, perf_d_2=None, perf_d_3=None, perf_d_4=None, title=None, save_fp=None,
                                    save_raw_data=None, annotation=None, pdf=True, include_mean=False):

    if perf_d_2 is None:
        if isinstance(perf_d_1, dict):
            perf_df = pd.DataFrame(perf_d_1).fillna(0)
            perf_df.columns = [i.replace(' Cancer', '') for i in perf_df.columns]
            if include_mean:
                perf_df['Mean'] = perf_df.apply(lambda x: np.average(a=x, weights=[1, 7, 12, 46, 54, 76, 133]), axis=1)
                perf_df.sort_values('Mean', ascending=False, inplace=True)
            else:
                perf_df.sort_values('Pan', ascending=False, inplace=True)
        else:
            perf_df = perf_d_1
        if save_raw_data is not None:
            perf_df.to_csv(save_raw_data)
        h = sns.heatmap(perf_df, cmap="RdYlBu", vmin=0, vmax=100, square=False, linewidth=0.3, annot=True,
                            cbar_kws={'label': 'Average Precision'})
        h.set_xticklabels(perf_df.columns, rotation=30, ha='right')

    else:
        if (perf_d_3 is None) and (perf_d_4 is None):
            fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.5))
        else:
            fig, axs = plt.subplots(2, 2, figsize=(8.5, 7))
        for i, coord, perf_ in zip(np.arange(0, 4),[[0, 0], [0, 1], [1, 0], [1, 1]],
                                   [perf_d_1, perf_d_2, perf_d_3, perf_d_4]):
            if isinstance(perf_, dict):
                perf_df = pd.DataFrame(perf_).fillna(0)
                perf_df.columns = [i.replace(' Cancer', '') for i in perf_df.columns]
                if include_mean:
                    perf_df['Mean'] = perf_df.apply(lambda x: np.average(a=x, weights=[1, 7, 12, 46, 54, 76, 133]), axis=1)
                    perf_df.sort_values('Mean', ascending=False, inplace=True)
                else:
                    perf_df.sort_values('Pan', ascending=False, inplace=True)
            else:
                perf_df = perf_
            if save_raw_data is not None:
                tmp = save_raw_data.split('.')
                tmp.insert(-1, f"df{i}")
                perf_df.to_csv(''.join(tmp))
            h = sns.heatmap(perf_df, cmap="RdYlBu", vmin=0, vmax=100, square=False,
                            linewidth=0.3, annot=True, ax=axs[coord[0], coord[1]], annot_kws={"fontsize": 6},
                            xticklabels=True, yticklabels=True)
            h.figure.axes[-1].set_ylabel('Average Precision', size=6)
            h.set_xticklabels(perf_df.columns, rotation=30, ha='right', fontsize=6)
            h.set_yticklabels(perf_df.index, fontsize=6)
            h.figure.axes[-1].tick_params(labelsize=6)

            if annotation is not None:
                print(annotation[i])
                axs[coord[0], coord[1]].annotate(annotation[i], xy=(-0.45, 1), xycoords='axes fraction', fontsize=8,
                                horizontalalignment='left', verticalalignment='top')

    if title is not None:
        plt.title(title)
    plt.tight_layout()

    if save_fp is not None:
        save_fp = save_fp+".pdf" if pdf else save_fp
        plt.savefig(save_fp, dpi=800)
        plt.close()
    else:
        plt.show()


def plot_similarity_clustermap(cls, pos_d, disease, save_fp=None):
    pos_df_overlap = pd.DataFrame(data=np.zeros((len(cls), len(cls))), columns=cls,
                                  index=cls)
    for a, b in product(cls, repeat=2):
        pos_df_overlap.loc[a, b] = len(set(pos_d[a]) & set(pos_d[b])) / len(set(pos_d[b]))
    pos_df_overlap.index = [i.split('_')[0] for i in pos_df_overlap.index]
    pos_df_overlap.columns = [i.split('_')[0] for i in pos_df_overlap.columns]

    h2 = sns.clustermap(pos_df_overlap, vmin=0, vmax=1)
    plt.title(f"Similarity in positives: {disease}")
    plt.tight_layout()
    if save_fp is not None:
        plt.savefig(f"depmaponly_figures/intermediates_similarity_clustermap_{disease.replace(' ', '_')}")
        plt.close()
    else:
        plt.show()


def read_gmt_file(fp, nw_obj):
    genes_per_DB = {}
    with open(fp) as f:
        lines = f.readlines()
        for line in lines:
            temp = line.strip('\n').split('\t')
            genes_per_DB[temp[0]] = set(gene for gene in temp[2:]) & set(nw_obj.node_names)
    return genes_per_DB


def calculate_gsea(pathway_d, gene_list, M=None, pval=0.05, gene_to_include=None, prerank=False, outdir=None,
                   processes=12, max_size=500, min_size=15):
    """

    :param pathway_d: dict of pathways to calculate enrichment for
    :param gene_list: ranked df if prerank=True, else normal list of genes
    :param M:
    :param pval:
    :param gene_to_include:
    :param prerank:
    :param outdir:
    :return:
    """
    if prerank:
        res = gp.prerank(rnk=gene_list, gene_sets=pathway_d, outdir=outdir, processes=processes,
                         min_size=min_size, max_size=max_size).results
        gsea = {}
        if gene_to_include is not None:
            gsea[gene_to_include] = []
            for path, d_ in res.items():
                if (d_["es"] > 0) & (d_['fdr'] < 0.05) & (gene_to_include in d_["ledge_genes"]):
                    gsea[gene_to_include].append(' '.join(path.split('_')[1:]).lower().capitalize())
            return res, gsea
        else:
            return res

    else:

        pval_l = []
        path_list = []
        for pathway, members in pathway_d.items():
            hypergeom_reactome = hypergeom(M=M, n=len(members), N=len(gene_list))
            overlap = set(members) & gene_list
            pval_l.append(hypergeom_reactome.sf(len(overlap) - 1))
            path_list.append(pathway)
        reject, qvals, _, _ = multipletests(pval_l, pval, method='fdr_bh')
        significant_pathways = dict(zip(np.array(path_list)[reject], qvals[reject]))
        df = pd.Series(significant_pathways,
                         index=significant_pathways.keys()).sort_values().to_frame(name='FDR q-value')

        if gene_to_include is not None:
            gsea = []
            for path in df.index:
                if gene_to_include in pathway_d[path]:
                    gsea.append(' '.join(path.split('_')[1:]).lower().capitalize())
            return df, gsea
        else:
            return df


def plot_boxplot_target_retrieval(prc_targets_d, disease, methods_significant,
                                  save_fp=None, method_name_map=None, save_raw_data=None,
                                  pdf=True, title=None):
    sns_to_plot_df_ = pd.DataFrame(prc_targets_d[disease]).transpose()
    methods_sign = [sns_to_plot_df_.columns.get_loc(i) for i in methods_significant]

    if method_name_map is not None:
        sns_to_plot_df_.columns = [method_name_map[n]
                                   if n in method_name_map else n
                                   for n in sns_to_plot_df_.columns]
    sns_to_plot_df_.columns = [i.replace('-', '\n') for i in sns_to_plot_df_.columns]

    if save_raw_data is not None:
        sns_to_plot_df_.to_csv(save_raw_data)

    df = sns_to_plot_df_.melt()

    fig, ax = plt.subplots(figsize=(8.5, 4))
    bx = sns.boxplot(x="variable", y="value", data=df, ax=ax, palette='pastel')
    bx.spines['top'].set_visible(False)
    bx.spines['right'].set_visible(False)
    # plt.tick_params(axis="both", which="major", length=5)
    bx.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # plt.title(f"Boxplot showing drug target retrievel across cell lines for {disease}")
    for i in methods_sign:
        ax.get_xticklabels()[i].set_color("white")
        ax.get_xticklabels()[i].set_weight("bold")
        ax.get_xticklabels()[i].set_bbox(dict(facecolor="red", alpha=0.9))
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_ylabel("Percentage of benchmark targets retrieved", fontsize=6)
    ax.set_xlabel("Method", fontsize=6)
    plt.tight_layout()

    if title:
        plt.title(title)
    if save_fp is not None:
        save_fp = save_fp + ".pdf" if pdf else save_fp
        plt.savefig(save_fp, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_distribution_drug_sens(drug2targets, all_pos, all_interm, subset_drug_sens, dis_df,
                                title=None, save_fp=None, save_raw_data=None, pdf=True, ax=None,
                                annotation=None):

    drugs_with_atleast_one_TP_as_target = {k: v for k, v in drug2targets.items() if
                                           (set(v) & all_pos and not set(v) & all_interm)}
    pos_no_targets = len(set.union(*[set(i) for i in drugs_with_atleast_one_TP_as_target.values()]))

    for k, v in drugs_with_atleast_one_TP_as_target.items():
        assert not set(v) & all_interm

    drugs_with_atleast_one_INTERMEDIARY_as_target = {k: v for k, v in drug2targets.items() if
                                                     (set(v) & all_interm and not set(v) & all_pos)}
    interm_no_targets = len(set.union(*[set(i) for i in drugs_with_atleast_one_INTERMEDIARY_as_target.values()]))

    for k, v in drugs_with_atleast_one_INTERMEDIARY_as_target.items():
        assert not set(v) & all_pos

    neg_no_targets = len(set([drug2targets[v][0] for v in drug2targets if v not in
                              set(drugs_with_atleast_one_TP_as_target.keys()) |
                              set(drugs_with_atleast_one_INTERMEDIARY_as_target.keys())]) & set(dis_df.columns))

    essential = subset_drug_sens[drugs_with_atleast_one_TP_as_target.keys()].values.reshape(-1, 1)
    intermediary_essential = subset_drug_sens[
        drugs_with_atleast_one_INTERMEDIARY_as_target.keys()].values.reshape(-1, 1)
    non_essential = subset_drug_sens.drop(set(drugs_with_atleast_one_TP_as_target.keys()) |
                                          set(drugs_with_atleast_one_INTERMEDIARY_as_target.keys()),
                                          axis=1).values.reshape(-1, 1)

    k_stat, k_pval = kruskal(essential.ravel(), intermediary_essential.ravel(), non_essential.ravel(), nan_policy='omit')
    print(k_stat, k_pval)
    no_targets = {'pos': pos_no_targets, 'interm': interm_no_targets, 'neg': neg_no_targets}

    plot_df = pd.concat(
        [pd.DataFrame(np.hstack((essential, np.array(
            [f"Extremely Strong\nDependency\n{no_targets['pos']} Targets"] * essential.shape[0]).reshape(-1, 1))),
                      columns=['drug_sens_profile', 'label']),
         pd.DataFrame(np.hstack((intermediary_essential,
                                 np.array([f"Intermediary\nDependency\n{no_targets['interm']} Targets"] *
                                          intermediary_essential.shape[0]).reshape(
                                     -1, 1))),
                      columns=['drug_sens_profile', 'label']),
         pd.DataFrame(
             np.hstack((non_essential, np.array(
                 [f"Extremely Weak\nDependency\n{no_targets['neg']} Targets"] * non_essential.shape[0]).reshape(-1, 1))),
             columns=['drug_sens_profile', 'label'])]).reset_index(drop=True)
    plot_df.loc[:, 'drug_sens_profile'] = plot_df.drug_sens_profile.astype(float)

    if save_raw_data is not None:
        plot_df.to_csv(save_raw_data)

    if ax is None:
        fig, ax = plt.subplots(figsize=((8, 6)))
        sns.violinplot(x="label", y="drug_sens_profile", data=plot_df, ax=ax, palette=sns.color_palette('colorblind'))
        ax.set_ylabel("Drug Sensitivity Score", fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.tick_params(axis="both", which="major", length=5)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.label.set_visible(False)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
    else:
        sns.violinplot(x="label", y="drug_sens_profile", data=plot_df, ax=ax, palette=sns.color_palette('colorblind'))
        ax.set_ylabel("Drug Sensitivity Score", fontsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.tick_params(axis="both", which="major", length=5)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.label.set_visible(False)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

    if annotation is not None:
        print(annotation)
        ax.annotate(annotation, xy=(-0.2, 1), xycoords='axes fraction', fontsize=6,
                    horizontalalignment='left', verticalalignment='top')

    if title is not None:
        ax.set_title(title, fontsize=8)

    if ax is None:
        plt.tight_layout()
        if save_fp is not None:
            save_fp = save_fp + ".pdf" if pdf else save_fp
            plt.savefig(save_fp, dpi=300)
        else:
            plt.show()


def plot_network_characteristic(single_feature_dict, largest_cc, title, target_overlap_cc=None, save_fp=None):

    tmp_df = pd.DataFrame({"gene": list(single_feature_dict.keys()),
                           "value": list(single_feature_dict.values())})
    tmp_df.sort_values("value", inplace=True)
    tmp_df.value = tmp_df.value.astype(float)
    if target_overlap_cc is not None:
        clrs = ['red' if (t in target_overlap_cc) else 'grey' for t in largest_cc.nodes]
    else:
        clrs = ["grey"]*largest_cc.size()
    hp = sns.barplot(x="gene", y="value", data=tmp_df, palette=clrs)
    hp.set(xticklabels=[])
    hp.set(xlabel=None)
    plt.title(title)
    if save_fp is not None:
        plt.savefig(save_fp)
        plt.close()
    else:
        plt.show()


def overlapping_coeff(network_a, network_b):
    G = nx.Graph()
    G.add_edges_from(set(network_a.edges) & set(network_b.edges))
    ol = G.size()
    small_network = network_a.size() if network_a.size() < network_b.size() else network_b.size()
    return ol / small_network


def jaccard_sim(network_a, network_b):
    G = nx.Graph()
    G.add_edges_from(set(network_a.edges) & set(network_b.edges))
    ol = G.size()
    return ol / (network_a.size() + network_b.size() - ol)


def plot_degree_distribution(network_G, title, save_fp=None):
    degree_sequence = sorted([d for n, d in network_G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")

    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    if save_fp is not None:
        plt.savefig(save_fp)
        plt.close()
    else:
        plt.show()


def eval_baselineEMBS_usingDLP(nee, scoresheet, baseline_method, save_preds=None,
                               predefined_embeddings=None, freeze_embs=False):
    """
    Experiment to test other embedding methods not integrated in the library.
    """
    print('Evaluating Embedding methods...')

    BASE_PATH = '/'.join(os.getcwd().split('/')[:-1])

    # Set edge embedding methods
    edge_embedding_methods = ['average', 'hadamard', 'weighted_l1', 'weighted_l2']

    # Evaluate non OpenNE method
    # -------------------------------
    # Set the methods
    methods_other = [f"DLP-weighted-l2-{baseline_method}"]

    # Set the method types
    method_type = ['e2e']

    # Set the commands
    commands_other = [
        "python %s/DeepLinkPrediction/DeepLinkPrediction/main_DLP.py --inputgraph {} "
        "--tr_e {} --tr_e_labels {} --te_e {} --te_e_labels {} --tr_pred {} --te_pred {} --dimension {} "
        f"--epochs 5 --merge_method weighted_l2  --validation_ratio 0.2 --predifined_embs {predefined_embeddings}"
        f"{' --freeze_embs' if freeze_embs else ''}" % BASE_PATH]

    # Set delimiters for the in and out files required by the methods
    input_delim = [',']
    output_delim = [',']

    for i in range(len(methods_other)):
        # Evaluate the method
        results = nee.evaluate_cmd(method_name=methods_other[i], method_type=method_type[i], command=commands_other[i],
                                   edge_embedding_methods=edge_embedding_methods,
                                   input_delim=input_delim[i], output_delim=output_delim[i], save_preds=save_preds)
        # Log the list of results
        scoresheet.log_results(results)


def get_cell_line_interaction_probs(total_df, cls):
    cl_prob_df = pd.DataFrame(data=np.zeros((len(cls), len(cls)), dtype=float), index=cls, columns=cls)

    ix = total_df[total_df.TestEdges_A.isin(cls) & total_df.TestEdges_B.isin(cls)]
    for i, row in ix.iterrows():
        cl_prob_df.loc[row.TestEdges_A, row.TestEdges_B] = row.Mean
    return cl_prob_df


def generate_top100_d(methods, common_cls, disease, npr_ppi, npr_dep, ppi_scaffold, topK,
                      all_cls, DW_compar="", embs_freeze="", emb_dim=128, total_df=None):
    top100_d = {}
    for method in methods:
        print(f"Method: {method}")
        top100_d[method] = {}
        if method == "original":
            total_df = pd.read_csv(f"depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}.csv",
                                   header=0, index_col=0)
        else:
            if total_df is None:
                total_df = pd.read_pickle(glob.glob("/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark/"
                                                    f"onlyDEPALL_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb{emb_dim}/"
                                                    f"{disease.replace(' ', '_')}{DW_compar}{embs_freeze}/"
                                                    f"{method}*/"
                                                    f"full_df_allruns_{disease.replace(' ', '_')}_emb{emb_dim}.pickle")[0])
        for i, cl in enumerate(common_cls):
            print(f"{i + 1}/{len(common_cls)} - {cl}")
            top100 = set()
            if method == "original":
                top100_d[method][cl] = get_topK_intermediaries(total_df, cl, top100, topK, original=True, top=True,
                                                               cl_list=all_cls)
            else:
                top100_d[method][cl] = get_topK_intermediaries(total_df, cl, top100, topK, original=False,
                                                               top=True, cl_list=all_cls)
    return top100_d


def closeness_centrality_(cl, top100_df, G):
    tmp_list_closeness_c = {}
    for method in top100_df.columns:
        # print(method)
        tmp_list_closeness_c[method] = [nx.closeness_centrality(G, u=n) for n in top100_df.loc[cl, method]]

    bxplot_df_2 = pd.DataFrame(tmp_list_closeness_c)
    bxplot_df_2.to_pickle(f"drug_sensitivity_data/"
                          f"networkx_characteristics/Lung Cancer/closeness_centrality_dict_{cl}.pickle")
    bxplot_df_2.columns = [i.replace("-opene", "") for i in bxplot_df_2.columns]
    bxplot_df_2 = bxplot_df_2.melt()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="variable", y="value", data=bxplot_df_2, palette='pastel', ax=ax)
    plt.savefig(f"drug_sensitivity_data/networkx_characteristics/Lung Cancer/closeness_centrality_{cl}")
    plt.close()
    return tmp_list_closeness_c


def calc_closeness_centrality_across_cls(G, cls, top100_df):
    cls = list(cls)
    if len(cls) == 39:
        k = 13
        i_ = int(39 / k)
    elif len(cls) == 88:
        k = 11
        i_ = int(88 / k)
    else:
        k = len(cls)
        i_ = 1

    if top100_df.columns.isin(["DLP-weighted-l1", "DLP-weighted-l2", "DLP-average"]).sum() == 3:
        top100_df.drop(columns=["DLP-weighted-l1", "DLP-weighted-l2", "DLP-average"], inplace=True)
    path_func = delayed(closeness_centrality_)
    return_dict = {}
    for i in range(i_):
        # print(f"{i + 1}/{i_}")
        cls_d = Parallel(n_jobs=k, verbose=True, backend='multiprocessing')(
        path_func(cl, top100_df, G) for n, cl in enumerate(cls[i * k:(i * k) + k]))

        return_dict.update({cl: update for cl, update in zip(cls[i*k:(i*k)+k], cls_d)})

    return return_dict


def shortest_paths_across_celllines(cls, end, top100_df_deep_copy, G):
    shortest_path_cl = {}
    for i, cl in enumerate(cls):
        # cl, i = "CAL12T_LUNG", 0
        print(f"{i + 1}/{len(cls)} - {cl}")
        tmp_dict_5 = {}
        for target in end.loc[cl]:
            # target = "KIF11"
            tmp_l = list(nx.algorithms.shortest_paths.all_shortest_paths(G, source=cl, target=target))
            tmp_dict_5[target] = set.union(*[set(i) for i in tmp_l])

        if tmp_dict_5:
            shortest_path_cl[cl] = set.union(*list(tmp_dict_5.values())) - set(tmp_dict_5.keys())
        else:
            shortest_path_cl[cl] = set()

    for cl in top100_df_deep_copy.index:
        top100_df_deep_copy.loc[cl] = top100_df_deep_copy.loc[cl].apply(lambda x: len(x & shortest_path_cl[cl]))

    return None


def prioritization_maarten(cell_gene_mat, query_gene, ppi_mat=None):
    query2cell_vec = cell_gene_mat[query_gene].values
    scores = np.matmul(query2cell_vec, cell_gene_mat.values)

    if ppi_mat is None:
        return pd.Series(scores, index=cell_gene_mat.columns).sort_values(ascending=False)

    else:

        cell_scores = pd.Series(scores, index=cell_gene_mat.columns).sort_values(ascending=False)
        total_score = ppi_mat[query_gene] + cell_scores.loc[ppi_mat.index]/query2cell_vec.shape[0]

        return total_score.sort_values(ascending=False)


def plot_subplot_col(plot_df, plot_df2, extra_information, axes=None, label_rot=0, add_annotation=False,
                     annotation=None):

    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.5, 4.5), sharey=True)

    b = sns.barplot(x="Target", y="no. sensitive\ncell lines retrieved", hue='Method',
                    data=plot_df, ax=axes[0], palette='colorblind')
    b.legend_.remove()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].xaxis.label.set_visible(False)
    axes[0].set_ylim(0, 65)
    axes[0].tick_params(axis='x', labelsize=6)
    axes[0].tick_params(axis='y', labelsize=6)
    axes[0].set_ylabel("no. sensitive\ncell lines retrieved", fontsize=6)
    axes[0].set_title("Targets retrieved by DLP-DeepWalk AND DepMap", fontsize=8)
    labels = [l.get_text() for l in axes[0].get_xticklabels()]

    extra_information1 = extra_information.loc[labels]
    print(extra_information1.shape)

    ax3b = axes[0].twinx()  # instantiate a second axes that shares the same x-axis

    xs = np.sort([patch.xy[0] for i, patch in enumerate(b.patches)])
    xs = xs[1::2]

    color = 'tab:blue'
    ax3b.set_ylabel('                 # Neighbors', color="k", fontsize=6)  # we already handled the x-label with ax1
    # ax3b.plot(xs, extra_information["noPos"], color="green", label="Positive cell lines", marker="o")
    ax3b.plot(xs, extra_information1["firstOnb"], color='k', marker="s", label="Positive neighbors")
    ax3b.tick_params(axis='y', labelcolor="k", labelsize=6)
    ax3b.set_ylim(-100, 100)
    ax3b.spines['top'].set_visible(False)
    ax3b.spines['right'].set_visible(False)
    ax3b.spines['bottom'].set_visible(False)

    if add_annotation:
        for x, y in zip(xs, extra_information1["firstOnb"].values):
            ax3b.annotate(str(y), (x - 0.2, y + 10), fontsize=6)

    b3 = sns.barplot(x="Target", y="no. sensitive\ncell lines retrieved", hue='Method',
                     data=plot_df2, ax=axes[1], palette='colorblind')
    b3.legend_.remove()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].xaxis.label.set_visible(False)
    axes[1].tick_params(axis='x', labelsize=6)
    axes[1].tick_params(axis='y', labelsize=6)
    axes[1].set_ylabel("no. sensitive\ncell lines retrieved", fontsize=6)
    axes[1].set_title("Targets retrieved by DLP-DeepWalk OR DepMap", fontsize=8)
    labels = [l.get_text() for l in axes[1].get_xticklabels()]

    extra_information2 = extra_information.loc[labels]
    print(extra_information2.shape)

    ax3b = axes[1].twinx()  # instantiate a second axes that shares the same x-axis

    xs = np.sort([patch.xy[0] for i, patch in enumerate(b3.patches)])
    xs = xs[1::2]

    color = 'tab:blue'
    ax3b.set_ylabel('                 # Neighbors', color="k", fontsize=6)  # we already handled the x-label with ax1
    # ax3b.plot(xs, extra_information["noPos"], color="green", label="Positive cell lines", marker="o")
    ax3b.plot(xs, extra_information2["firstOnb"], color='k', marker="s", label="Positive neighbors")
    ax3b.tick_params(axis='y', labelcolor="k", labelsize=6)
    ax3b.set_ylim(-100, 100)
    ax3b.spines['top'].set_visible(False)
    ax3b.spines['right'].set_visible(False)
    ax3b.spines['bottom'].set_visible(False)

    if add_annotation:
        for x, y in zip(xs, extra_information2["firstOnb"].values):
            ax3b.annotate(str(y), (x - 0.2, y + 10), fontsize=6)

    if annotation is not None:
        axes[0].annotate(annotation[0], xy=(-0.2, 1), xycoords='axes fraction', fontsize=6,
                         horizontalalignment='left', verticalalignment='top')
        axes[1].annotate(annotation[1], xy=(-0.2, 1), xycoords='axes fraction', fontsize=6,
                         horizontalalignment='left', verticalalignment='top')

    for tick in axes[1].get_xticklabels():
        tick.set_rotation(label_rot)

    for tick in axes[0].get_xticklabels():
        tick.set_rotation(label_rot)

    return ax3b


def check_equality_dict(d1, d2):
    assert len(d1) == len(d2), "ERROR length dicts are not the same"
    assert len(set(d1.keys()) & set(d2.keys())) == len(d1), "ERROR different keys"
    for k, v in d1.items():
        assert sorted(v) == sorted(d2[k]), "ERROR, different values per key"


def best_edgeembed_graphsage(perf_dict):
    perf_df = pd.DataFrame(perf_dict)*100
    return_dict = {}
    return_dict["ap"] = perf_df.mean().iloc[[0, 4, 8, 12]].max()
    return_dict["f1"] = perf_df.mean().iloc[[1, 5, 9, 13]].max()
    return_dict["acc"] = perf_df.mean().iloc[[2, 6, 10, 14]].max()
    return_dict["auc"] = perf_df.mean().iloc[[3, 7, 11, 15]].max()

    max_embed = perf_df.mean().iloc[[0, 4, 8, 12]].idxmax()
    tmp = pd.DataFrame(perf_df[max_embed]).transpose()
    tmp.index = ["GraphSAGE"]
    tmp.columns = ["run0", "run1", "run2"]
    return_dict["run_ap"] = tmp

    return return_dict


def evaluate_dw(ppi_scaffold, disease, screening, train_edges, train_labels, test_edges, test_labels,
                train_edges_val, train_labels_val, test_edges_val, test_labels_val, repeat, npr_ppi, npr_dep,
                pos_thresh="", save_embs=True, save_preds=None, train_ratio=100):
    traintest_split = EvalSplit()
    traintest_split.set_splits(train_E=train_edges[np.where(train_labels == 1)[0]],
                               train_E_false=train_edges[np.where(train_labels == 0)[0]],
                               test_E=test_edges[np.where(test_labels == 1)[0]],
                               test_E_false=test_edges[np.where(test_labels == 0)[0]],
                               directed=False, nw_name=f'{ppi_scaffold}_dependencies',
                               TG=None, split_id=repeat, split_alg='dlp', owa=True, verbose=True)

    trainvalid_split = EvalSplit()
    trainvalid_split.set_splits(train_E=train_edges_val[np.where(train_labels_val == 1)[0]],
                                train_E_false=train_edges_val[np.where(train_labels_val == 0)[0]],
                                test_E=test_edges_val[np.where(test_labels_val == 1)[0]],
                                test_E_false=test_edges_val[np.where(test_labels_val == 0)[0]],
                                directed=False, nw_name=f'{ppi_scaffold}_dependencies',
                                TG=None, split_id=repeat, split_alg='dlp', owa=True, verbose=True)

    nee = LPEvaluator(traintest_split=traintest_split, trainvalid_split=trainvalid_split, dim=128)

    # Set edge embedding methods
    edge_embedding_methods = ['weighted_l2']

    # Evaluate methods from OpenNE
    # ----------------------------
    # Set the methods
    methods = ['deepwalk-opene']

    # Set the commands
    commands = [
        'python -m openne --method deepWalk --graph-format edgelist --epochs 10 --number-walks 10 --walk-length 80']

    # For each method evaluate
    for i in range(0, len(methods)):
        command = commands[i] + " --input {} --output {} --representation-size {}"
        if methods[i] == 'deepwalk-opene':
            save_embs = f"{methods[i]}_{disease.replace(' ', '_')}_{ppi_scaffold}_embsize128_{train_ratio}percent{screening}{pos_thresh}" \
                        f"_nprPPI{npr_ppi}_nprDEP{npr_dep}" \
                if save_embs else None
        else:
            save_embs = None
        print(f"\nSave embs outide EvalNE {save_embs}\n")
        results = nee.evaluate_cmd(method_name=methods[i], method_type='ne', command=command,
                                   edge_embedding_methods=edge_embedding_methods, input_delim=' ', output_delim=' ',
                                   tune_params=None, verbose=True, save_preds=save_preds,
                                   save_embs=save_embs)
        # Log the list of results
        # scoresheet.log_results(results)


def transductive_performance(d_of_df):
    performance = {}
    for k, v in d_of_df.items():
        performance[k] = []
        for repeat in range(3):
            performance[k].append(average_precision_score(v["label"], v[f"rep{repeat}"]))

    performance_df = pd.DataFrame(performance)
    performance_df.index = ["rep0", "rep1", "rep2"]
    return performance_df


def construct_EvalNE_splits(train_edges, train_labels, train_edges_val, train_labels_val, test_edges, test_labels,
                            test_edges_val, test_labels_val, ppi_scaffold, repeat, dim):
    traintest_split = EvalSplit()
    traintest_split.set_splits(train_E=train_edges[np.where(train_labels == 1)[0]],
                               train_E_false=train_edges[np.where(train_labels == 0)[0]],
                               test_E=test_edges[np.where(test_labels == 1)[0]],
                               test_E_false=test_edges[np.where(test_labels == 0)[0]],
                               directed=False, nw_name=f'{ppi_scaffold}_dependencies',
                               TG=None, split_id=repeat, split_alg='dlp', owa=True, verbose=True)

    trainvalid_split = EvalSplit()
    trainvalid_split.set_splits(train_E=train_edges_val[np.where(train_labels_val == 1)[0]],
                                train_E_false=train_edges_val[np.where(train_labels_val == 0)[0]],
                                test_E=test_edges_val[np.where(test_labels_val == 1)[0]],
                                test_E_false=test_edges_val[np.where(test_labels_val == 0)[0]],
                                directed=False, nw_name=f'{ppi_scaffold}_dependencies',
                                TG=None, split_id=repeat, split_alg='dlp', owa=True, verbose=True)

    return LPEvaluator(traintest_split=traintest_split, trainvalid_split=trainvalid_split, dim=dim)


def eval_other(nee, scoresheet, npr_ppi, npr_dep, disease, BASE_PATH, train_ratio, save_preds=None, save_embs=True,
               emb_size=128, ppi_scaffold="", screening="", pos_thresh=None):
    """
    Experiment to test other embedding methods not integrated in the library.
    """
    print('Evaluating Embedding methods...')

    # Set edge embedding methods
    edge_embedding_methods = ['average', 'hadamard', 'weighted_l1', 'weighted_l2']

    # Evaluate methods from OpenNE
    # ----------------------------
    # Set the methods
    methods = ['deepwalk-opene', 'line-opene', 'n2v-opene', 'grarep-opene']
    # methods = ['line-opene', 'n2v-opene', 'grarep-opene']

    # Set the commands
    commands = [
        'python -m openne --method deepWalk --graph-format edgelist --epochs 10 --number-walks 10 --walk-length 80',
        'python -m openne --method line --epochs 10 --order 3',
        'python -m openne --method node2vec --walk-length 80 --number-walks 10 --epochs 10',
        'python -m openne --method grarep --graph-format edgelist --epochs 10 --kstep 2'
        ]
    # commands = [
    #     'python -m openne --method line --epochs 10 --order 3',
    #     'python -m openne --method node2vec --walk-length 80 --number-walks 10 --epochs 10',
    #     'python -m openne --method grarep --graph-format edgelist --epochs 10 --kstep 2'
    # ]

    # For each method evaluate
    for i in range(0, len(methods)):
        command = commands[i] + " --input {} --output {} --representation-size {}"
        if methods[i] == 'deepwalk-opene':
            save_embs = f"{methods[i]}_{disease.replace(' ', '_')}_{ppi_scaffold}_embsize{emb_size}_{train_ratio}percent{screening}{pos_thresh}" \
                        f"_nprPPI{npr_ppi}_nprDEP{npr_dep}" \
                if save_embs else None
        else:
            save_embs = None
        print(f"\nSave embs outide EvalNE {save_embs}\n")
        results = nee.evaluate_cmd(method_name=methods[i], method_type='ne', command=command,
                                   edge_embedding_methods=edge_embedding_methods, input_delim=' ', output_delim=' ',
                                   tune_params=None, verbose=True, save_preds=save_preds,
                                   save_embs=save_embs)
        # Log the list of results
        scoresheet.log_results(results)

    # Evaluate non OpenNE method
    # -------------------------------
    # Set the methods
    methods_other = ['AROPE', 'VERSE', 'DLP-hadamard']#, 'metapath2vec++']
    # methods_other = ['metapath2vec++']

    # Set the method types
    method_type = ['e2e', 'ne', 'e2e']#, 'ne']
    # method_type = ['ne']

    # Set the commands
    commands_other = ["/home/bioit/pstrybol/anaconda3/envs/DepMap_DeepLinkPrediction_Benchmark_py2/bin/python "
                      f"{BASE_PATH}/AROPE/main_Arope.py "
                      "--inputgraph {} --tr_e {} --te_e {} --tr_pred {} --te_pred {} --dimension {} "
                      "--order 3 --weights '[1, 0.1, 0.01]'",

                      f"python {BASE_PATH}/verse/main_VERSE.py "
                      "--input {} --output {} --dimension {} --undirected --alpha 0.85 --threads 40",

                      f"python {BASE_PATH}/DeepLinkPrediction/DeepLinkPrediction/main_DLP.py "
                      "--inputgraph {} --tr_e {} --tr_e_labels {} --te_e {} --te_e_labels {} --tr_pred {} --te_pred {} "
                      "--dimension {} --epochs 5 --merge_method hadamard  --validation_ratio 0.2"]#,

                      #'../code_metapath2vec/metapath2vec -min-count 1 -iter 20 -samples 100 -train {} -output {} -size {}']
    # commands_other = [
    #     '../code_metapath2vec/metapath2vec -min-count 1 -iter 20 -samples 100 -train {} -output {} -size {}']

    # Set delimiters for the in and out files required by the methods
    input_delim = [',', ',', ',']#, ' ']
    # input_delim = [' ']
    output_delim = [',', ',', ',']#, ' ']
    # output_delim = [' ']

    for i in range(len(methods_other)):
        # Evaluate the method
        results = nee.evaluate_cmd(method_name=methods_other[i], method_type=method_type[i],
                                   command=commands_other[i],
                                   edge_embedding_methods=edge_embedding_methods,
                                   input_delim=input_delim[i], output_delim=output_delim[i], save_preds=save_preds)
        # Log the list of results
        scoresheet.log_results(results)


def eval_baselines(nee, directed, scoresheet, save_preds=None):
    """
    Experiment to test the baselines.
    """
    print('Evaluating baselines...')

    # Set the baselines
    methods = ['common_neighbours', 'jaccard_coefficient', 'adamic_adar_index', 'resource_allocation_index',
               'preferential_attachment', 'random_prediction', 'all_baselines']

    # Evaluate baseline methods
    for method in methods:
        if directed:
            result = nee.evaluate_baseline(method=method, neighbourhood="in", save_preds=save_preds)
            scoresheet.log_results(result)
            result = nee.evaluate_baseline(method=method, neighbourhood="out", save_preds=save_preds)
            scoresheet.log_results(result)
        else:
            result = nee.evaluate_baseline(method=method, save_preds=save_preds)
            scoresheet.log_results(result)


def construct_combined_df_DLPdeepwalk(BASE_PATH, repeats, ppi_scaffold, screening, pan_nw_obj, ppi_obj):
    tot_df_l = []
    for repeat in range(repeats):
        total_df = pd.read_pickle(f"{BASE_PATH}/EvalNE_pancancer_target_prediction/{ppi_scaffold}{screening}/"
                                  f"PanCancer_revised_setting_DLP-DeepWalk_predictions_STRING_{repeat}.pickle")
        total_df.columns = ['TestEdges_A', 'TestEdges_B', f'predictions_rep{repeat}', 'labels']
        total_df[['TestEdges_A', 'TestEdges_B']] = total_df[['TestEdges_A', 'TestEdges_B']]. \
            applymap(lambda x: pan_nw_obj.int2gene[x])
        tot_df_l.append(total_df[total_df['TestEdges_B'].isin(ppi_obj.node_names)])

    all_df_v2 = reduce(lambda left, right: pd.merge(left, right, on=['TestEdges_A', 'TestEdges_B']), tot_df_l)
    all_df_v2['Mean'] = all_df_v2[[i for i in list(all_df_v2) if i.startswith('predictions')]].mean(axis=1)

    return all_df_v2