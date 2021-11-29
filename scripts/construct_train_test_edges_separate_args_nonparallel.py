from DeepLinkPrediction.utils import *
from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--disease', required=True, type=str)
parser.add_argument('--screening', required=True, type=str)
parser.add_argument('--ppi_scaffold', required=True, type=str)
parser.add_argument('--npr_ppi', required=True, type=int)
parser.add_argument('--npr_dep', required=True, type=int)
parser.add_argument('--pos_thresh', required=True, type=float)
parser.add_argument('--neg_thresh', required=True, type=float)
parser.add_argument('--train_ratio', required=True, type=float)
parser.add_argument('--val_ratio', required=True, type=float)
parser.add_argument('--transductive', required=False, action='store_true')
parser.add_argument('--bin', required=False, type=str)
parser.add_argument('--gene', required=False, type=str)
parser.add_argument('--demo', action="store_true")
args = parser.parse_args()

if args.demo:
    assert os.getcwd().split('/')[-1] == "demo", "Wrong working directory"
    BASE_PATH = os.getcwd()
else:
    if os.getcwd().endswith("DepMap_DeepLinkPrediction_Benchmark"):
        BASE_PATH = '/'.join(os.getcwd().split('/'))
    else:
        BASE_PATH = '/'.join(os.getcwd().split('/')[:-1])
print(BASE_PATH)

# Arguments #
PPI_SCAFFOLD_LOC = f"{BASE_PATH}/ppi_network_scaffolds/"
ppi_scaffold = args.ppi_scaffold
# ppi_scaffold = "STRING"
print(ppi_scaffold+'\n')
train_ratio = args.train_ratio
# train_ratio = 0.8
val_ratio = args.val_ratio
# val_ratio = 0.8
disease = args.disease
screening = '' if args.screening == 'rnai' else '_crispr'
# screening = ''
negative_dep_threshold = args.neg_thresh
# negative_dep_threshold = -0.5
positive_dep_threshold = args.pos_thresh
pos_thresh_str = f"_pos{str(args.pos_thresh).replace('.', '_')}" if args.pos_thresh != -1.5 else ""
bin_ = args.bin
gene = args.gene
# positive_dep_threshold = -1.5

if args.transductive:
    dis_df = pd.read_csv(f"{BASE_PATH}/depmap_specific_cancer_df/{bin_}_{gene}_{ppi_scaffold}_{disease.replace(' ','_')}{screening}"
                         f"{pos_thresh_str}.csv", header=0, index_col=0)
else:
    dis_df = pd.read_csv(
        f"{BASE_PATH}/depmap_specific_cancer_df/{ppi_scaffold}_{disease.replace(' ', '_')}{screening}"
        f"{pos_thresh_str}.csv", header=0, index_col=0)

npr_ppi = args.npr_ppi
# npr_ppi = 5
npr_dep = args.npr_dep
# npr_dep = 3

# Make sure output directory exists, if not, make one
if args.transductive:
    out_path = f"{BASE_PATH}/LP_train_test_splits{screening}/" \
               f"{bin_}_{gene}_{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}{pos_thresh_str}/" \
               f"{disease.replace(' ', '_')}"
else:
    out_path = f"{BASE_PATH}/LP_train_test_splits{screening}/" \
               f"{ppi_scaffold}_split_nprPPI{npr_ppi}_nprDEP{npr_dep}{pos_thresh_str}/" \
               f"{disease.replace(' ','_')}"
print(out_path)

if not os.path.isdir(out_path):
    os.makedirs(out_path)

# Number separate train/test sets to construct
num_splits = 3

# Sanity check on mapped node names to indices
ndex_nw_obj = read_ppi_scaffold(ppi_scaffold, PPI_SCAFFOLD_LOC)
gene2int, int2gene = updated_gene2int(ndex_nw_obj.gene2int, dis_df)
with open(f'{out_path}/gene2int.pickle', 'wb') as handle:
    pickle.dump(gene2int, handle, protocol=pickle.HIGHEST_PROTOCOL)

if args.transductive:
    print("heterogeneous network transductive")
    heterogeneous_network = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                                        f"{bin_}_{gene}_{ppi_scaffold}_{disease.replace(' ', '_')}_dependencies{screening}"
                                        f"{pos_thresh_str}.csv")
else:
    heterogeneous_network = pd.read_csv(f"{BASE_PATH}/heterogeneous_networks/"
                                        f"{ppi_scaffold}_{disease.replace(' ','_')}_dependencies{screening}"
                                        f"{pos_thresh_str}.csv")

heterogeneous_network_obj = UndirectedInteractionNetwork(heterogeneous_network)
assert gene2int == heterogeneous_network_obj.gene2int, "ERROR INDEX MISMATCH"

# Complete interaction set specific for cell line - gene dependencies incl. cell line - cell line interactions
complete_preds = pd.DataFrame([[a, b] for a, b in product([heterogeneous_network_obj.gene2int[i] for i in dis_df.index],
                                         [heterogeneous_network_obj.gene2int[i] for i in dis_df.columns])])

# complete_preds_2 = pd.DataFrame([[a, b] for a, b in product([heterogeneous_network_obj.gene2int[i] for i in dis_df.index],
#                                          [heterogeneous_network_obj.gene2int[i] for i in dis_df.index])])
# complete_preds = pd.concat([complete_preds_1, complete_preds_2])
write_h5py(os.path.join(out_path, "complete_predset_without_cl2cl.hdf5"), data=complete_preds.values)

# To save RAM
del complete_preds

# Complete interaction set specific for PPI interactions
# complete_preds = np.array(list(product(ndex_nw_obj.nodes, ndex_nw_obj.nodes)))
# write_h5py(os.path.join(out_path, "complete_predset_PPI.hdf5"), data=complete_preds)

for split_id in range(num_splits):
    print(f"Num split {split_id}")
    X_train_, X_val_, X_test_, Y_train_, Y_val_, Y_test_  = ndex_nw_obj.getTrainTestData(neg_pos_ratio=npr_ppi,
                                                                                         train_ratio=train_ratio,
                                                                                         train_validation_ratio=val_ratio,
                                                                                         return_summary=False)

    print("PPI Split Done")
    negs, negs_arr_train, negs_arr_val, negs_arr_test, pos, \
    pos_arr_train, pos_arr_val, pos_arr_test, \
    intermediate, intermediate_arr_train, intermediate_arr_val, \
    intermediate_arr_test = generate_traintest_dependencies(dis_df,
                                                            threshold_neg=negative_dep_threshold,
                                                            threshold_pos=positive_dep_threshold,
                                                            npr=npr_dep,
                                                            gene2int=gene2int,
                                                            train_test_ratio=train_ratio,
                                                            train_validaiton_ratio=val_ratio,
                                                            exclude_negs={gene})
    print("Dependency Split Done")



    # Consciously AVOID training on intermediates => watch out for cell lines with only 1 dependency, this will result
    # in a cell line not occuring in the training set. Only cell lines with at least 3 positives are considered
    X_train, X_val, X_test, y_train, y_val, y_test = construct_combined_traintest(pos_arr_train=pos_arr_train,
                                                                                  negs_arr_train=negs_arr_train,
                                                                                  X_train_=X_train_, Y_train_=Y_train_,
                                                                                  pos_arr_val=pos_arr_val,
                                                                                  negs_arr_val=negs_arr_val,
                                                                                  X_val_=X_val_, Y_val_=Y_val_,
                                                                                  pos_arr_test=pos_arr_test,
                                                                                  negs_arr_test=negs_arr_test,
                                                                                  X_test_=X_test_, Y_test_=Y_test_)

    nodes_lost = set(heterogeneous_network_obj.nodes) - set(np.unique(X_train[np.where(y_train == 1)].ravel()))
    assert not nodes_lost, \
        f"ERROR NDOES LOST DURING SPLIT, {[heterogeneous_network_obj.int2gene[i] for i in nodes_lost]}"

    # Store the computed interaction sets to a file
    filenames = (os.path.join(out_path, "trE_{}.hdf5".format(split_id)),
                 os.path.join(out_path, "label_trE_{}.hdf5".format(split_id)),
                 os.path.join(out_path, "teE_{}.hdf5".format(split_id)),
                 os.path.join(out_path, "label_teE_{}.hdf5".format(split_id)),
                 os.path.join(out_path, "trE_val_{}.hdf5".format(split_id)),
                 os.path.join(out_path, "label_trE_val_{}.hdf5".format(split_id)),
                 os.path.join(out_path, "teE_val_{}.hdf5".format(split_id)),
                 os.path.join(out_path, "label_teE_val_{}.hdf5".format(split_id)),
                 os.path.join(out_path, "test_all_cls_{}.hdf5".format(split_id)),
                 os.path.join(out_path, "label_test_all_cls_{}.hdf5".format(split_id)),
                 os.path.join(out_path, "test_all_ppis_{}.hdf5".format(split_id)),
                 os.path.join(out_path, "labeltest_all_ppis_{}.hdf5".format(split_id))
                 )

    # IMPORTANT, EVALNE REQUIRES 4 SETS : TRAIN_VAL, TEST_VAL, TRAIN_TEST, TEST_TEST
    # WHICH IS WHY I CONCATENATE X_TRAIN WITH X_VAL TO MAKE TRAIN_TEST
    X_train_VAL = X_train
    X_test_VAL = X_val
    X_train_TEST = np.vstack((X_train, X_val))
    X_test_TEST = X_test

    y_train_VAL = y_train
    y_test_VAL = y_val
    y_train_TEST = np.hstack((y_train, y_val))
    y_test_TEST = y_test

    # Construct cell line test sets separately, don't include intermediaries, these are not shown during training
    cell_line_test_neg = np.vstack((pos_arr_test, negs_arr_test))
    cell_line_test_neg_labels = np.hstack((np.ones(pos_arr_test.shape[0]), np.zeros(negs_arr_test.shape[0])))

    # Save the splits in different files
    for fn, data in zip(filenames, [X_train_TEST, y_train_TEST, X_test_TEST, y_test_TEST, X_train_VAL, y_train_VAL,
                                    X_test_VAL, y_test_VAL, cell_line_test_neg, cell_line_test_neg_labels, X_test_,
                                    Y_test_]):
        # for fn, data in zip(filenames[-2:], [X_test_, Y_test_]):
        if data is not None:
            write_h5py(fn, data=data)
