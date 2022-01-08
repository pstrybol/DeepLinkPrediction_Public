from DeepLinkPrediction.utils import *
from DeepLinkPrediction.DLembedder import DLembedder, TimingCallback
from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
from evalne.utils.preprocess import read_node_embeddings
import numpy as np
import argparse
import os

"""
python /home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark/main_DLP.py
 --inputgraph ./edgelist.tmp
  --tr_e ./tmp_tr_e.tmp 
  --tr_e_labels ./tmp_tr_e_labels.tmp 
  --te_e ./tmp_te_e.tmp
  --te_e_labels ./tmp_te_e_labels.tmp 
  --tr_pred ./tmp_tr_out.tmp 
  --te_pred ./tmp_te_out.tmp 
  --dimension 10 --epochs 5 --merge_method hadamard  --validation_ratio 0.2
"""

def parse_args():
    """ Parses DLP arguments."""

    parser = argparse.ArgumentParser(description="Run DLP.")

    parser.add_argument('--inputgraph', nargs='?',
                        default='heterogeneous_networks/PCNet_dependencies.csv',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?',
                        default='.',
                        help='Path where the embeddings will be stored.')

    parser.add_argument('--tr_e', nargs='?', default=None,
                        help='Path of the input train edges. Default None (only returns embeddings)')

    parser.add_argument('--tr_e_labels', nargs='?', default=None,
                        help='Path of the input train edges LABELS. Default None (only returns embeddings)')

    parser.add_argument('--tr_pred', nargs='?', default='tr_pred_.csv',
                        help='Path where the train predictions will be stored.')

    parser.add_argument('--te_e', nargs='?', default=None,
                        help='Path of the input test edges.')

    parser.add_argument('--te_e_labels', nargs='?', default=None,
                        help='Path of the input test edges LABELS.')

    parser.add_argument('--te_pred', nargs='?', default='te_pred.csv',
                        help='Path where the test predictions will be stored.')

    parser.add_argument('--test_performance', default=None,
                        help='Path where the test performance will be stored.')

    parser.add_argument('--dimension', type=int, default=128,
                        help='Embedding dimension. Default is 10.')

    parser.add_argument('--nodes_per_layer', type=list, default=[32, 32, 1],
                        help='How many hidden nodes to use per layer. Default is [32, 32, 1]')

    parser.add_argument('--activations_per_layer', type=list, default=['relu', 'relu', 'sigmoid'],
                        help='How many hidden nodes to use per layer. Default is [`relu`, `relu`, `sigmoid`]')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Percentage of nodes to drop during training.')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs. Default is 10.')

    parser.add_argument('--validation_ratio', type=float, default=0.2, # either give data or set to 0
                        help='Percentage of training data to use for validation during training. Default is 0')

    parser.add_argument('--seed', type=int, default=6,
                        help='Training seed used for randomization initialization')

    parser.add_argument('--optimizer', default='adam',
                        help='Optimizer to be used. Options are `adam`, `lbfgs`, `grad_desc` Default is `adam`.')

    parser.add_argument('--metrics', default='binary_accuracy',
                        help='Metrics to optimize during training. Default is binary accuracy.')

    parser.add_argument('--loss', default='binary_crossentropy',
                        help='Loss function to optimize. Default is binary cross entropy.')

    parser.add_argument('--custom_loss', type=bool, default=False,
                        help='Whether to use a loss function or not.')

    parser.add_argument('--merge_method', default='weighted_l1',
                        help='Merge method to construct edge embeddings from node embeddings.'
                             ' Default is ABS-DIF. Options are `flatten`, `average`, `hadamard`, `difference`.')

    parser.add_argument('--verbose', type=int, default=2,
                        help='Verbosity of the keras DL model.')

    parser.add_argument('--allow_nans', type=bool, default=False,
                        help='If, in case of multiloss learning, to allow Nans during training.')

    parser.add_argument('--model_save', type=str, default='DLP_results/DLP_models/',
                        help='Path of model to use when saving the model.')

    parser.add_argument('--delimiter', default=',',
                        help='The delimiter used to separate the edgelist.')

    parser.add_argument("--predifined_embs", type=str, default=None,
                        help="Predefine the embedding layer, i.e. set it fixed, recommended for prediction only")

    parser.add_argument("--freeze_embs", default=False,
                        help="Freeze embedding layer.", action='store_true')
    # parser.add_argument('--repeat', default=',', type=int,
    #                     help='')

    return parser.parse_args()


def main(tr_e, tr_e_labels, inputgraph, merge_method, output=None, te_e=None, tr_pred=None, te_pred=None,
         dimension=128, nodes_per_layer=None, activations_per_layer=None,
         dropout=0.2, seed=6, validation_ratio=0.2, epochs=5, allow_nans=False,
         metrics='binary_accuracy', loss='binary_crossentropy',
         freeze_embs=False, verbose=2, predifined_embs=None, delimiter=',', return_predictions=False,
         additional_test_set=None):

    nodes_per_layer = [32, 32, 1] if nodes_per_layer is None else nodes_per_layer
    activations_per_layer = ['relu', 'relu', 'sigmoid'] if activations_per_layer is None else activations_per_layer

    print("\n\t Reading training edges\n")
    if isinstance(tr_e, str) | isinstance(tr_e_labels, str):
        train_edges = np.loadtxt(tr_e, delimiter=delimiter, dtype=int)
        train_labels = np.loadtxt(tr_e_labels, delimiter=delimiter, dtype=int)
    else:
        train_edges = tr_e
        train_labels = tr_e_labels

    if isinstance(inputgraph, str):
        nw_df = pd.read_csv(inputgraph, header=0, sep=delimiter)
        nw_obj = UndirectedInteractionNetwork(nw_df)
    else:
        nw_obj = inputgraph

    dl_net = DLembedder(nw_obj.N_nodes, dimension, nodes_per_layer=nodes_per_layer,
                        activations_per_layer=activations_per_layer, int2genedict=None,
                        dropout=dropout, merge_method=merge_method, random_state=seed)

    dl_net.counter = 0

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
    time_cb = TimingCallback()

    print("\n\t Fitting the model \n")
    if predifined_embs is not None:
        embs_d = read_node_embeddings(predifined_embs, nodes=nw_obj.nodes,
                                      embed_dim=dimension, method_name="deepwalk-opene", delimiter=' ')
        embs_df = pd.DataFrame(embs_d).transpose()
        embs_df.index = embs_df.index.astype(int)
        embs_df.sort_index(inplace=True)

        _ = dl_net.fit(train_edges, train_labels, validation_split=validation_ratio, callbacks=[earlyStopping, time_cb],
                       n_epochs=epochs, verbose=verbose, allow_nans=allow_nans, metrics=metrics,
                       loss=loss, predefined_embeddings=embs_df.values, freeze_embedding=freeze_embs)
    else:
        _ = dl_net.fit(train_edges, train_labels, validation_split=validation_ratio,
                       callbacks=[earlyStopping, time_cb],
                       n_epochs=epochs, verbose=verbose, allow_nans=allow_nans, metrics=metrics,
                       loss=loss, predefined_embeddings=None, freeze_embedding=freeze_embs)

    if tr_pred is not None:
        np.savetxt(tr_pred, dl_net.predict_proba(train_edges), delimiter=delimiter)

    if output is not None:
        print("\n\t Saving embeddings \n")
        try:
            os.makedirs(output)
        except FileExistsError:
            print("Folder already exists")

        np.savetxt(output + '.emb', dl_net.getEmbeddings(), delimiter=delimiter)

    if te_e is not None:
        print("\n\t Reading test edges\n")
        if isinstance(te_e, str):
            test_edges = np.loadtxt(te_e, delimiter=delimiter, dtype=int)
        else:
            test_edges = te_e

        predictions = dl_net.predict_proba(test_edges)
        np.savetxt(te_pred, predictions, delimiter=delimiter)

        if additional_test_set is not None:
            additional_predictions = dl_net.predict_proba(test_edges)
            if return_predictions:
                return predictions, additional_predictions

        if return_predictions:
            return predictions
    else:
        return dl_net

def main_cli(args):
    main(tr_e=args.tr_e, tr_e_labels=args.tr_e_labels, tr_pred=args.tr_pred, te_e=args.te_e, te_pred=args.te_pred, inputgraph=args.inputgraph,
         merge_method=args.merge_method, output=args.output,
         dimension=args.dimension, nodes_per_layer=args.nodes_per_layer, activations_per_layer=args.activations_per_layer,
         dropout=args.dropout, seed=args.seed, validation_ratio=args.validation_ratio, epochs=args.epochs, allow_nans=args.allow_nans,
         metrics=args.metrics, loss=args.loss,
         freeze_embs=args.freeze_embs, verbose=args.verbose, predifined_embs=args.predifined_embs, delimiter=args.delimiter)


if __name__ == "__main__":
    args = parse_args()
    main_cli(args)
