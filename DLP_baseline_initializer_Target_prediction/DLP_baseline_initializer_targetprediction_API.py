from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
from DeepLinkPrediction.utils import read_h5py, eval_baselineEMBS_usingDLP
from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.split import EvalSplit
from evalne.evaluation.score import Scoresheet
import pandas as pd
import numpy as np
import argparse
import random
import glob
import re
import os

assert os.getcwd().split('/')[-1] == "DLP_baseline_initializer_Target_prediction", "Wrong working directory"
BASE_PATH = '/'.join(os.getcwd().split('/')[:-1])

parser = argparse.ArgumentParser()
parser.add_argument('--baseline_method', required=True, type=str)
parser.add_argument('--ppi_scaffold', required=True, type=str)
parser.add_argument('--disease', required=True, type=str)
parser.add_argument('--screening', required=True, type=str)
parser.add_argument('--npr_ppi', required=True, type=int)
parser.add_argument('--npr_dep', required=True, type=int)
parser.add_argument('--emb_dim', required=True, type=int)
parser.add_argument('--pos_thresh', required=False, type=str)
args = parser.parse_args()

method = args.baseline_method
print(f"\n\t {method}\n")
disease = args.disease
npr_dep = args.npr_dep
npr_ppi = args.npr_ppi
ppi_scaffold = args.ppi_scaffold
emb_dim = args.emb_dim
train_ratio = 100
screening = '' if args.screening == 'rnai' else '_crispr'
pos_thresh = f"_pos{args.pos_thresh.replace('.', '_')}" if args.pos_thresh else ""

emb_path = BASE_PATH + f"/EvalNE_CellLine_specific_total_predictions/" \
                       f"{method}_{disease.replace(' ', '_')}_{ppi_scaffold}_embsize{emb_dim}" \
                       f"_{train_ratio}percent{screening}{pos_thresh}/"
print(emb_path)

traintest_loc = BASE_PATH + f"/LP_train_test_splits{screening}/" \
                            f"{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}{pos_thresh}/{disease.replace(' ','_')}/"
all_file_loc = glob.glob(traintest_loc+'*')

heterogeneous_network = pd.read_csv(BASE_PATH+f"/heterogeneous_networks/"
                                              f"{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies{screening}{pos_thresh}.csv")
heterogeneous_network_obj = UndirectedInteractionNetwork(heterogeneous_network)

CL_TEST = read_h5py(BASE_PATH+f"/LP_train_test_splits{screening}/"
                    f"{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}{pos_thresh}/{disease.replace(' ','_')}/"
                    f"complete_predset_without_cl2cl.hdf5")
assert pd.DataFrame(CL_TEST).shape == pd.DataFrame(CL_TEST).drop_duplicates().shape
CL_TEST_LABELS = np.array([random.randint(0, 1) for _ in range(CL_TEST.shape[0])])

# Create a Scoresheet to store the results
scoresheet = Scoresheet(tr_te='test')

for repeat in range(3):
    print(repeat)

    print("\n\tTraining on 100%\n")
    train_edges_ = read_h5py(list(filter(re.compile(rf'(.*/trE_{repeat}.hdf5)').match, all_file_loc))[0])
    train_labels_ = read_h5py(list(filter(re.compile(rf'(.*/label_trE_{repeat}.hdf5)').match, all_file_loc))[0])
    test_edges_tocombine_with_train = read_h5py(
        list(filter(re.compile(rf'(.*/teE_{repeat}.hdf5)').match, all_file_loc))[0])
    test_labels_tocombine_with_train = read_h5py(
        list(filter(re.compile(rf'(.*/label_teE_{repeat}.hdf5)').match, all_file_loc))[0])

    train_edges = np.vstack((train_edges_, test_edges_tocombine_with_train))
    train_labels = np.hstack((train_labels_, test_labels_tocombine_with_train))

    test_edges = CL_TEST
    test_labels = CL_TEST_LABELS

    # Validation train and test set
    train_edges_val = read_h5py(list(filter(re.compile(rf'(.*/trE_val_{repeat}.hdf5)').match, all_file_loc))[0])
    train_labels_val = read_h5py(list(filter(re.compile(rf'(.*/label_trE_val_{repeat}.hdf5)').match, all_file_loc))[0])
    test_edges_val = read_h5py(list(filter(re.compile(rf'(.*/teE_val_{repeat}.hdf5)').match, all_file_loc))[0])
    test_labels_val = read_h5py(list(filter(re.compile(rf'(.*/label_teE_val_{repeat}.hdf5)').match, all_file_loc))[0])

    # Create an evaluator and generate train/test edge split
    traintest_split = EvalSplit()
    traintest_split.set_splits(train_E=train_edges[np.where(train_labels == 1)[0]],
                               train_E_false=train_edges[np.where(train_labels == 0)[0]],
                               test_E=test_edges[np.where(test_labels == 1)[0]],
                               test_E_false=test_edges[np.where(test_labels == 0)[0]],
                               directed=False, nw_name=f'{ppi_scaffold}_dependencies',
                               TG=None, split_id=repeat, split_alg='dlp', owa=True, verbose=True)

    # Even though this will never be evaluated it still needs to be defined?
    trainvalid_split = EvalSplit()
    trainvalid_split.set_splits(train_E=train_edges_val[np.where(train_labels_val == 1)[0]],
                                train_E_false=train_edges_val[np.where(train_labels_val == 0)[0]],
                                test_E=test_edges_val[np.where(test_labels_val == 1)[0]],
                                test_E_false=test_edges_val[np.where(test_labels_val == 0)[0]],
                                directed=False, nw_name=f'{ppi_scaffold}_dependencies',
                                TG=None, split_id=repeat, split_alg='dlp', owa=True, verbose=True)

    nee = LPEvaluator(traintest_split=traintest_split, trainvalid_split=trainvalid_split, dim=emb_dim)

    # Evaluate other NE methods
    save_preds_fp_cell_line = BASE_PATH+f"/onlyDEPALL_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb{emb_dim}{screening}{pos_thresh}/" \
                                        f"{disease.replace(' ','_')}_{train_ratio}percent"
    save_preds_fp_ppi = BASE_PATH + f"/PPIALL_5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_{ppi_scaffold}_emb{emb_dim}{screening}/" \
                                    f"{disease.replace(' ', '_')}_{train_ratio}percent"
    try:
        os.makedirs(save_preds_fp_ppi)
    except FileExistsError:
        print("folder {} already exists".format(save_preds_fp_cell_line))

    print("\n\tSaving Predictions\n")
    eval_baselineEMBS_usingDLP(nee, scoresheet, save_preds=save_preds_fp_cell_line,
                               predefined_embeddings=emb_path + f"emb_{method}_{repeat}.tmp",
                               freeze_embs=False, baseline_method=method)


