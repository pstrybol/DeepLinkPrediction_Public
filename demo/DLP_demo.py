import time
start = time.time()
from sklearn.metrics import roc_auc_score, average_precision_score
from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
from DeepLinkPrediction.utils import read_h5py
from DeepLinkPrediction.main_DLP import main
from evalne.utils import util
import pandas as pd
import numpy as np
import glob
import re
import os

assert os.getcwd().split('/')[-1] == "demo", "Wrong working directory"

disease = "Bladder Cancer"
ppi_scaffold = "PID"
npr_ppi = 5
npr_dep = 3
pos_thresh = -1.5
neg_thresh = -0.5
train_ratio = 0.8
val_ratio = 0.8
heterogeneous_network = pd.read_csv(f"heterogeneous_networks/"
                                    f"{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies.csv")
heterogeneous_network_obj = UndirectedInteractionNetwork(heterogeneous_network)
# Step 1: construct the train and test set for a particular disease and funtional interaction scaffold
command_train_test_split = f"python ../scripts/construct_train_test_edges_separate_args.py " \
                           f"--disease '{disease}' --ppi_scaffold '{ppi_scaffold}' --npr_ppi {npr_ppi} --npr_dep {npr_dep} " \
                           f"--pos_thresh {pos_thresh} --neg_thresh {neg_thresh} --train_ratio {train_ratio} " \
                           f"--val_ratio {val_ratio} --demo"

util.run(cmd=command_train_test_split, timeout=31536000-(time.time() - start), verbose=True)

# Step 2: run the DeepLinkPrediction model on the constructed train/test sets
all_file_loc = glob.glob(f"LP_train_test_splits/{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}/"
                         f"{disease.replace(' ','_')}/*")

tr_pred = f"DLP_predictions/{ppi_scaffold}/{disease.replace(' ','_')}/train"
te_pred = f"DLP_predictions/{ppi_scaffold}/{disease.replace(' ','_')}/test"

try:
    os.makedirs(tr_pred)
except FileExistsError:
    print(f"Directory {tr_pred} already exists")

try:
    os.makedirs(te_pred)
except FileExistsError:
    print(f"Directory {te_pred} already exists")

inputgraph = f"heterogeneous_networks/{ppi_scaffold}_{disease.replace(' ','_')}_dependencies.csv"
merge_method = "hadamard"
output = "."

tot_prediction_df = {}
performance = {"auc": [], "ap": [], "mean": []}

general_performance = False
for repeat in range(3):
    train_edges = read_h5py(list(filter(re.compile(rf'(.*/trE_{repeat}.hdf5)').match, all_file_loc))[0])
    train_labels = read_h5py(list(filter(re.compile(rf'(.*/label_trE_{repeat}.hdf5)').match, all_file_loc))[0])
    if general_performance:
        test_edges = read_h5py(list(filter(re.compile(rf'(.*/teE_{repeat}.hdf5)').match, all_file_loc))[0])
        test_labels = read_h5py(list(filter(re.compile(rf'(.*/label_teE_{repeat}.hdf5)').match, all_file_loc))[0])
    else: # this will evaluate dependency specific performance
        test_edges = read_h5py(list(filter(re.compile(rf'(.*/test_all_cls_{repeat}.hdf5)').match, all_file_loc))[0])
        test_labels = read_h5py(list(filter(re.compile(rf'(.*/label_test_all_cls_{repeat}.hdf5)').match, all_file_loc))[0])

    test_predictions = main(tr_e=train_edges, tr_e_labels=train_labels, te_e=test_edges,
                                             tr_pred=f"{tr_pred}/train_repeat{repeat}.txt",
                                             te_pred=f"{te_pred}/test_repeat{repeat}.txt", inputgraph=inputgraph,
                                             merge_method=merge_method, output=output, dimension=128,
                                             nodes_per_layer=None, activations_per_layer=None, dropout=0.2, seed=6,
                                             validation_ratio=0.2, epochs=5, allow_nans=False,
                                             metrics='binary_accuracy', loss='binary_crossentropy', freeze_embs=False,
                                             verbose=2, predifined_embs=None, delimiter=',', return_predictions=True)

    performance["auc"].append(roc_auc_score(test_labels, test_predictions))
    performance["ap"].append(average_precision_score(test_labels, test_predictions))

performance["mean"].append(np.mean(performance["auc"]))
performance["mean"].append(np.mean(performance["ap"]))

print(f"{ppi_scaffold} - {disease} - Mean AUC = {performance['mean'][0]} - Mean AP = {performance['mean'][1]}")

# Step 3: run DLP on a new test set containing all possible combinations between genes and cell lines in the network
# Note that we need to retrain the model since we are now training on 100% instead of 80%

test_edges = read_h5py(f"LP_train_test_splits/"
                       f"{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}/{disease.replace(' ','_')}/"
                       f"complete_predset_with_cl2cl.hdf5")

total_predictions = {}
for repeat in range(3):
    train_edges_ = read_h5py(list(filter(re.compile(rf'(.*/trE_{repeat}.hdf5)').match, all_file_loc))[0])
    train_labels_ = read_h5py(list(filter(re.compile(rf'(.*/label_trE_{repeat}.hdf5)').match, all_file_loc))[0])
    test_edges_tocombine_with_train = read_h5py(list(filter(re.compile(rf'(.*/teE_{repeat}.hdf5)').match, all_file_loc))[0])
    test_labels_tocombine_with_train = read_h5py(list(filter(re.compile(rf'(.*/label_teE_{repeat}.hdf5)').match, all_file_loc))[0])

    # Now we train on 100%
    train_edges = np.vstack((train_edges_, test_edges_tocombine_with_train))
    train_labels = np.hstack((train_labels_, test_labels_tocombine_with_train))

    test_edges = test_edges # Note that no labels are required because we don't calculate performance

    total_predictions[f"run{repeat}"] = main(tr_e=train_edges, tr_e_labels=train_labels, te_e=test_edges,
                             tr_pred=f"{tr_pred}/train_repeat{repeat}.txt",
                             te_pred=f"{te_pred}/total_test_repeat{repeat}.txt", inputgraph=inputgraph,
                             merge_method=merge_method, output=output, dimension=128,
                             nodes_per_layer=None, activations_per_layer=None, dropout=0.2, seed=6,
                             validation_ratio=0.2, epochs=5, allow_nans=False,
                             metrics='binary_accuracy', loss='binary_crossentropy', freeze_embs=False,
                             verbose=2, predifined_embs=None, delimiter=',', return_predictions=True)

total_predictions_df = pd.DataFrame({"GeneA": test_edges[:, 0], "GeneB": test_edges[:, 1],
                                     "predictions": np.mean([total_predictions["run0"],
                                                             total_predictions["run1"],
                                                             total_predictions["run2"]], axis=0).ravel()})
total_predictions_df[["GeneA", "GeneB"]] = total_predictions_df[["GeneA", "GeneB"]].\
    applymap(lambda x: heterogeneous_network_obj.int2gene[x])

end = time.time()
print(f"Execution took {end-start} seconds")




