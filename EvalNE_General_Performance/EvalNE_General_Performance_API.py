import os
os.environ["OPENBLAS_NUM_THREADS"] = "24"
from DeepLinkPrediction.utils import read_h5py
from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.split import EvalSplit
from evalne.evaluation.score import Scoresheet
import re
import numpy as np
import glob
import argparse

assert os.getcwd().split('/')[-1] == "EvalNE_General_Performance", "Wrong working directory"
BASE_PATH = '/'.join(os.getcwd().split('/')[:-1])

parser = argparse.ArgumentParser()
parser.add_argument('--ppi_scaffold', required=True, type=str)
parser.add_argument('--disease', required=True, type=str)
parser.add_argument('--screening', required=True, type=str)
parser.add_argument('--npr_ppi', required=True, type=int)
parser.add_argument('--npr_dep', required=True, type=int)
parser.add_argument('--emb_dim', required=True, type=int)
parser.add_argument('--pos_thresh', required=False, type=str)
args = parser.parse_args()


def eval_other(nee, scoresheet, save_preds=None, save_embs=False, emb_size=128, ppi_scaffold="", screening="",
               pos_thresh=None):
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

    # Set the commands
    commands = [
        'python -m openne --method deepWalk --graph-format edgelist --epochs 10 --number-walks 10 --walk-length 80',
        'python -m openne --method line --epochs 10 --order 3',
        'python -m openne --method node2vec --walk-length 80 --number-walks 10 --epochs 10',
        'python -m openne --method grarep --graph-format edgelist --epochs 10 --kstep 2'
        ]

    # For each method evaluate
    for i in range(0, len(methods)):
        command = commands[i] + " --input {} --output {} --representation-size {}"
        save_embs = f"{methods[i]}_{disease.replace(' ', '_')}_{ppi_scaffold}_embsize{emb_size}_80percent{screening}{pos_thresh}"\
                    if save_embs else None
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
    methods_other = ['AROPE', 'VERSE', 'DLP-hadamard']
    # methods_other = ['metapath2vec++']

    # Set the method types
    method_type = ['e2e', 'ne', 'e2e']
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
                      "--dimension {} --epochs 5 --merge_method hadamard  --validation_ratio 0.2"]

    # commands_other = ['../code_metapath2vec/metapath2vec -min-count 1 -iter 20 -samples 100 -train {} -output {} -size {}']

    # Set delimiters for the in and out files required by the methods
    input_delim = [',', ',', ',']
    # input_delim = [' ']
    output_delim = [',', ',', ',']
    # output_delim = [' ']

    for i in range(len(methods_other)):
        # Evaluate the method
        results = nee.evaluate_cmd(method_name=methods_other[i], method_type=method_type[i],
                                   command=commands_other[i],
                                   edge_embedding_methods=edge_embedding_methods,
                                   input_delim=input_delim[i], output_delim=output_delim[i], save_preds=save_preds)
        # Log the list of results
        scoresheet.log_results(results)


def eval_baselines(nee, directed, scoresheet):
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
            result = nee.evaluate_baseline(method=method, neighbourhood="in")
            scoresheet.log_results(result)
            result = nee.evaluate_baseline(method=method, neighbourhood="out")
            scoresheet.log_results(result)
        else:
            result = nee.evaluate_baseline(method=method)
            scoresheet.log_results(result)
#######################################################################################################################

# Load the network and train/test edges
ppi_scaffold = args.ppi_scaffold
disease = args.disease
screening = '' if args.screening == 'rnai' else '_crispr'
pos_thresh_str = f"_pos{str(args.pos_thresh).replace('.', '_')}" if args.pos_thresh != -1.5 else ""

print(ppi_scaffold+'\t'+disease+'\n')
all_file_loc = glob.glob(f"{BASE_PATH}/"
                         f"LP_train_test_splits{screening}/{ppi_scaffold}_split_nprPPI{args.npr_ppi}_nprDEP{args.npr_dep}/"
                         f"{disease.replace(' ','_')}/*")

# Create output directory if not already present
if not os.path.isdir(f"{BASE_PATH}/"
                     f"General_Benchmark_Res{screening}/"
                     f"{ppi_scaffold}/{disease.replace(' ','_')}"):
    os.makedirs(f"{BASE_PATH}/"
                f"General_Benchmark_Res{screening}/"
                f"{ppi_scaffold}/{disease.replace(' ','_')}")

# Create a Scoresheet to store the test results
scoresheet = Scoresheet(tr_te='test')

for repeat in range(3):
    print('Repetition {} of experiment - TRAINED ON 80%'.format(repeat))

    # Train and actual test set
    train_edges = read_h5py(list(filter(re.compile(rf'(.*/trE_{repeat}.hdf5)').match, all_file_loc))[0])
    train_labels = read_h5py(list(filter(re.compile(rf'(.*/label_trE_{repeat}.hdf5)').match, all_file_loc))[0])
    test_edges = read_h5py(list(filter(re.compile(rf'(.*/teE_{repeat}.hdf5)').match, all_file_loc))[0])
    test_labels = read_h5py(list(filter(re.compile(rf'(.*/label_teE_{repeat}.hdf5)').match, all_file_loc))[0])

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

    trainvalid_split = EvalSplit()
    trainvalid_split.set_splits(train_E=train_edges_val[np.where(train_labels_val == 1)[0]],
                                train_E_false=train_edges_val[np.where(train_labels_val == 0)[0]],
                                test_E=test_edges_val[np.where(test_labels_val == 1)[0]],
                                test_E_false=test_edges_val[np.where(test_labels_val == 0)[0]],
                                directed=False, nw_name=f'{ppi_scaffold}_dependencies',
                                TG=None, split_id=repeat, split_alg='dlp', owa=True, verbose=True)

    nee = LPEvaluator(traintest_split=traintest_split, trainvalid_split=trainvalid_split, dim=args.emb_dim)

    # Path where to save output predictions.
    save_preds_fp = f"{BASE_PATH}/"\
                    f"PPIandDEP_5epochs02valShuffled_nprPPI{args.npr_ppi}_nprDEP{args.npr_dep}_{ppi_scaffold}/" \
                    f"{disease.replace(' ','_')}_emb{args.emb_dim}{screening}"
    try:
        os.makedirs(save_preds_fp)
        print("folder '{}' created ".format(save_preds_fp))
    except FileExistsError:
        print("folder {} already exists".format(save_preds_fp))

    # Eval others
    eval_other(nee, scoresheet, save_preds=None, save_embs=False, emb_size=args.emb_dim, pos_thresh=pos_thresh_str)

    # Evaluate baselines
    eval_baselines(nee, directed=False, scoresheet=scoresheet)


scoresheet.write_all(f"{BASE_PATH}/"
                     f"General_Benchmark_Res{screening}/"
                     f"{ppi_scaffold}/{disease.replace(' ','_')}/"
                     f"5epochs02valShuffled_nprPPI{args.npr_ppi}_nprDEP{args.npr_dep}_"
                     f"{disease.replace(' ','_')}_emb{args.emb_dim}.txt", repeats='all')

scoresheet.write_pickle(f"{BASE_PATH}/"
                        f"General_Benchmark_Res{screening}/"
                        f"{ppi_scaffold}/{disease.replace(' ','_')}/"
                        f"5epochs02valShuffled_nprPPI{args.npr_ppi}_nprDEP{args.npr_dep}_"
                        f"{disease.replace(' ','_')}_emb{args.emb_dim}.pickle")
