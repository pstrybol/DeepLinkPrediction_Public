from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.split import EvalSplit
from evalne.evaluation.score import Scoresheet
from DeepLinkPrediction.utils import read_h5py, eval_baselineEMBS_usingDLP
import numpy as np
import glob
import re
import os
import argparse

assert os.getcwd().split('/')[-1] == "DLP_baseline_initializer_Performance", "Wrong working directory"
BASE_PATH = '/'.join(os.getcwd().split('/')[:-1])

parser = argparse.ArgumentParser()
parser.add_argument('--baseline_method', required=True, type=str)
parser.add_argument('--ppi_scaffold', required=True, type=str)
parser.add_argument('--disease', required=True, type=str)
parser.add_argument('--screening', required=True, type=str)
parser.add_argument('--npr_ppi', required=True, type=int)
parser.add_argument('--npr_dep', required=True, type=int)
parser.add_argument('--emb_dim', required=True, type=int)
parser.add_argument('--pos_thresh', required=False, type=float)
parser.add_argument('--general_performance', action="store_true") # if this argument is not passed -> dep performance is calculated
parser.add_argument('--ppi_performance', action="store_true") # if this argument is not passed -> dep performance is calculated
args = parser.parse_args()

method = args.baseline_method
print(f"\n\t {method}\n")
disease = args.disease
npr_dep = args.npr_dep
npr_ppi = args.npr_ppi
ppi_scaffold = args.ppi_scaffold
emb_dim = args.emb_dim
train_ratio = 80
screening = '' if args.screening == 'rnai' else '_crispr'
pos_thresh_str = f"_pos{str(args.pos_thresh).replace('.', '_')}" if args.pos_thresh != -1.5 else ""

if args.general_performance:
    print("\n\tCHECKING GENERAL PERFORMANCE\n")
elif args.ppi_performance:
    print("\n\tCHECKING PPI PERFORMANCE\n")
else:
    print("\n\tCHECKING DEPENDENCY SPECIFIC PERFORMANCE\n")

try:
    os.makedirs(BASE_PATH+f"/CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ','_')}{pos_thresh_str}/")
except:
    print("folder exists")

emb_path = BASE_PATH + f"/EvalNE_CellLine_specific_Performance/" \
                       f"{method}_{disease.replace(' ', '_')}_{ppi_scaffold}_embsize{emb_dim}_{train_ratio}percent{args.screening}{pos_thresh_str}" \
                       f"_nprPPI{npr_ppi}_nprDEP{npr_dep}/"
print(emb_path)

all_file_loc = glob.glob(BASE_PATH+f"/LP_train_test_splits{screening}/"
                                   f"{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}{pos_thresh_str}/"
                                   f"{disease.replace(' ','_')}/*")
# Create a Scoresheet to store the results
scoresheet = Scoresheet(tr_te='test')

for repeat in range(3):
    print(repeat)

    print("\n... Checking performance ... \n")
    train_edges = read_h5py(list(filter(re.compile(rf'(.*/trE_{repeat}.hdf5)').match, all_file_loc))[0])
    train_labels = read_h5py(list(filter(re.compile(rf'(.*/label_trE_{repeat}.hdf5)').match, all_file_loc))[0])

    # If we want to calculate performance
    if args.general_performance:
        test_edges = read_h5py(list(filter(re.compile(rf'(.*/teE_{repeat}.hdf5)').match, all_file_loc))[0])
        test_labels = read_h5py(list(filter(re.compile(rf'(.*/label_teE_{repeat}.hdf5)').match, all_file_loc))[0])
    elif args.ppi_performance:
        test_edges = read_h5py(list(filter(re.compile(rf'(.*/test_all_ppis_{repeat}.hdf5)').match, all_file_loc))[0])
        test_labels = read_h5py(
            list(filter(re.compile(rf'(.*/labeltest_all_ppis_{repeat}.hdf5)').match, all_file_loc))[0])
    else:
        test_edges = read_h5py(list(filter(re.compile(rf'(.*/test_all_cls_{repeat}.hdf5)').match, all_file_loc))[0])
        test_labels = read_h5py(
            list(filter(re.compile(rf'(.*/label_test_all_cls_{repeat}.hdf5)').match, all_file_loc))[0])


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

    # Evaluate DLP using a basline method to initialize the embedding layer
    eval_baselineEMBS_usingDLP(nee, scoresheet, save_preds=None,
                               predefined_embeddings=emb_path + f"emb_{method}_{repeat}.tmp",
                               freeze_embs=False, baseline_method=method)

if args.general_performance:
    scoresheet.write_all(BASE_PATH + f"/General_Benchmark_Res{screening}/"
                                     f"{ppi_scaffold}/{disease.replace(' ','_')}/"
                                     f"5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_"
                                     f"{disease.replace(' ','_')}_complete_metricPerofrmance_emb{emb_dim}_{method}"
                                     f"_pretrainedEMBS_{train_ratio}percent_embs_not_frozen.txt",
                         repeats='all')

    scoresheet.write_pickle(BASE_PATH + f"/General_Benchmark_Res{screening}/"
                                        f"{ppi_scaffold}/{disease.replace(' ','_')}/"
                                        f"5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_"
                                        f"{disease.replace(' ','_')}_complete_metricPerofrmance_emb{emb_dim}_{method}"
                                        f"_pretrainedEMBS_{train_ratio}percent_embs_not_frozen.pickle")
elif args.ppi_performance:
    scoresheet.write_all(f"{BASE_PATH}/"
                         f"PPI_Benchmark_Res{screening}/"
                         f"{ppi_scaffold}/{disease.replace(' ', '_')}/"
                         f"5epochs02valShuffled_nprPPI{args.npr_ppi}_nprDEP{args.npr_dep}_"
                         f"{disease.replace(' ', '_')}_complete_metricPerofrmance_emb{args.emb_dim}_DLP.txt", repeats='all')

    scoresheet.write_pickle(f"{BASE_PATH}/"
                            f"PPI_Benchmark_Res{screening}/"
                            f"{ppi_scaffold}/{disease.replace(' ', '_')}/"
                            f"5epochs02valShuffled_nprPPI{args.npr_ppi}_nprDEP{args.npr_dep}_"
                            f"{disease.replace(' ', '_')}_complete_metricPerofrmance_emb{args.emb_dim}_DLP.pickle")
else:
    scoresheet.write_all(BASE_PATH + f"/CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ','_')}/"
                                     f"5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_cellLinePerformance_emb{emb_dim}_{method}"
                                     f"_pretrainedEMBS_{train_ratio}percent_embs_not_frozen.txt",
                         repeats='all')

    scoresheet.write_pickle(BASE_PATH + f"/CellLine_Specific_Benchmark_Res{screening}/{ppi_scaffold}/{disease.replace(' ', '_')}/"
                                        f"5epochs02valShuffled_nprPPI{npr_ppi}_nprDEP{npr_dep}_cellLinePerformance_emb{emb_dim}_{method}"
                                        f"_pretrainedEMBS_{train_ratio}percent_embs_not_frozen.pickle")

