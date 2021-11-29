import pandas as pd
import numpy as np
import networkx as nx
import copy
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, confusion_matrix
from networkx import to_dict_of_lists
import matplotlib.pyplot as plt
from itertools import product


class Graph:
    '''
        Builds a graph from _path.
        _path: a path to a file containing "node_from node_to" edges (one per line)
    '''

    @classmethod
    def from_file(cls, path, colnames=('Gene1', 'Gene2'), sep=',',
                  header=0, column_index=None, keeplargestcomponent=False,
                  network_type='kegg', gene_id_type='symbol'):

        if network_type is None:
            network_df = pd.read_csv(path, sep=sep, header=header, low_memory=False, index_col=column_index)
            network_df = network_df[list(colnames)]
        elif network_type.lower() == 'kegg':
            network_df = pd.read_csv(path, sep='\t', header=0, dtype=str)[['from', 'to']]

        elif network_type.lower() == 'string':
            network_df = pd.read_csv(path, sep='\t', header=0)[['Gene1', 'Gene2']]

        elif network_type.lower() == 'biogrid':
            network_df = pd.read_csv(path, sep='\t', header=0)

            if gene_id_type.lower() == 'entrez':
                network_df = network_df[['Entrez Gene Interactor A', 'Entrez Gene Interactor B']]

            elif gene_id_type.lower() == 'symbol':
                network_df = network_df[['Official Symbol Interactor A', 'Official Symbol Interactor B']]

            else:
                raise IOError('gene_id_type not understood.'
                              'For Biogrid please specify entrez or symbol.')

        else:
            raise IOError('Network type not understood.'
                          'Please specify kegg, biogrid or reactome, or enter None for custom network type.')

        return cls(network_df, keeplargestcomponent=keeplargestcomponent)

    def __init__(self, interaction_df, colnames=None, verbose=True, keeplargestcomponent=False,
                 allow_self_connected=False):
        '''
        :param: interaction_df a pandas edgelist consisting of (at least two) columns,
        indicating the two nodes for each edge
        :param: colnames, the names of the columns that contain the nodes and optionally some edge attributes.
        The first two columns must indicate the nodes from the edgelsist
        '''

        def isinteger(x):
            try:
                return np.all(np.equal(np.mod(x, 1), 0))

            except:
                return False

        self.attr_names = None

        if colnames is not None:
            interaction_df = interaction_df[list(colnames)]
            if len(colnames) > 2:
                self.attr_names = colnames[2:]  # TODO this needs to be done better
        elif interaction_df.shape[1] == 2:
            interaction_df = interaction_df

        else:
            print('Continuing with %s and %s as columns for the nodes' % (interaction_df.columns.values[0],
                                                                          interaction_df.columns.values[1]))
            interaction_df = interaction_df.iloc[:, :2]

        interaction_df = interaction_df.drop_duplicates()
        self.interactions = interaction_df
        old_col_names = list(self.interactions.columns)
        self.interactions.rename(columns={old_col_names[0]: 'Gene_A', old_col_names[1]: 'Gene_B'}, inplace=True)

        if not allow_self_connected:
            self.interactions = self.interactions.loc[self.interactions.Gene_A != self.interactions.Gene_B]

        if isinteger(self.interactions.Gene_A.values):  # for integer nodes do numerical ordering of the node_names
            node_names = np.unique(self.interactions[['Gene_A', 'Gene_B']].values)
            self.interactions = self.interactions.astype(str)
            node_names = node_names.astype(str)

        else:
            self.interactions = self.interactions.astype(str)
            # node_names = np.unique(self.interactions[['Gene_A', 'Gene_B']].values)
            node_names = pd.unique(self.interactions[['Gene_A', 'Gene_B']].values.flatten())

        self.int2gene = {i: name for i, name in enumerate(node_names)}
        gene2int = self.gene2int

        self.interactions = self.interactions.applymap(lambda x: gene2int[x])
        self.nodes = np.array([gene2int[s] for s in node_names]).astype(np.int)

        self.embedding_dict = None
        if keeplargestcomponent:
            self.keepLargestComponent(verbose=verbose, inplace=True)

        if verbose:
            print('%d Nodes and %d interactions' % (len(self.nodes),
                                                    self.interactions.shape[0]))

    def deepcopy(self):
        return copy.deepcopy(self)

    @property
    def gene2int(self):
        return {v: k for k, v in self.int2gene.items()}

    @property
    def node_names(self):
        return np.array([self.int2gene[i] for i in range(self.N_nodes)])

    @property
    def N_nodes(self):
        return len(self.nodes)

    @property
    def N_interactions(self):
        return self.interactions.shape[0]

    def __contains__(self, gene):
        return gene in self.node_names

    def __repr__(self):
        return self.getInteractionNamed().__repr__()

    def __str__(self):
        return self.getInteractionNamed().__str__()

    def __len__(self):
        return self.N_nodes

    def __eq__(self, other):
        if isinstance(other, Graph):
            return self.interactions_as_set() == other.interactions_as_set()
        return NotImplemented

    def getInteractionNamed(self, return_both_directions=False):
        if return_both_directions:
            df = self.interactions.applymap(lambda x: self.int2gene[x])
            df2 = df.copy(deep=True).rename(columns={'Gene_B': 'Gene_A', 'Gene_A': 'Gene_B'})
            return pd.concat([df, df2], axis=0, ignore_index=True)
        else:
            return self.interactions.applymap(lambda x: self.int2gene[x])


    def makeSelfConnected(self, inplace=False):
        self_df = pd.DataFrame({'Gene_A': self.node_names, 'Gene_B': self.node_names})

        if inplace:
            self_df = self_df.applymap(lambda x: self.gene2int[x])
            self.interactions = pd.concat([self.interactions, self_df], ignore_index=True)

        else:
            new_df = pd.concat([self.getInteractionNamed(), self_df], ignore_index=True)
            return new_df


    def getAdjMatrix(self, sort='first', as_df=False):

        row_ids = list(self.interactions['Gene_A'])
        col_ids = list(self.interactions['Gene_B'])

        A = np.zeros((self.N_nodes, self.N_nodes), dtype=np.uint8)
        A[(row_ids, col_ids)] = 1

        if as_df:
            return pd.DataFrame(A, index=self.node_names, columns=self.node_names)
        else:
            return A, np.array(self.node_names)


    def getAdjDict(self, return_names=True):
        pass

    def getNOrderNeighbors(self, order=2, include_lower_order=True, gene_list=None):

        adj_dict = copy.deepcopy(self.getAdjDict())
        orig_dict = self.getAdjDict()

        if gene_list is not None:
            adj_dict = {k: v for k, v in adj_dict.items() if k in gene_list}

        for _ in range(order-1):
            adj_dict = getSecondOrderNeighbors(adj_dict, adj_dict0=orig_dict,
                                               incl_first_order=include_lower_order)
        return adj_dict

    def getDegreeDF(self, return_names=True, set_index=False):
        v, c = np.unique(self.interactions.values.flatten(), return_counts=True)
        if return_names:
            if set_index:
                return pd.DataFrame({'Gene': [self.int2gene[i] for i in v],
                                     'Count': c}, index=[self.int2gene[i] for i in v]).sort_values(by='Count',
                                                                                                   ascending=False,
                                                                                                   inplace=False)
            else:
                return pd.DataFrame({'Gene': [self.int2gene[i] for i in v],
                                     'Count': c}).sort_values(by='Count', ascending=False, inplace=False)
        else:
            if set_index:
                return pd.DataFrame({'Gene': v,
                                     'Count': c}, index=v).sort_values(by='Count', ascending=False, inplace=False)
            else:
                return pd.DataFrame({'Gene': v,
                                 'Count': c}).sort_values(by='Count', ascending=False, inplace=False)

    def visualize(self, return_large=False, gene_list=None, edge_df=None, show_labels=False,
                  node_colors=None, cmap='spectral', title=None,
                  color_scheme_nodes=('lightskyblue', 'tab:orange'),
                  color_scheme_edges=('gray', 'tab:green'), labels_dict=None,
                  filename=None, save_path=None):

        """ Visualize the graph
         gene_list = MUST be a list of lists
         labels_dict: a dictionary of dictionaries, containing the labels, fontsizes etc for each group of labels.

         example: {'group1': {'labels': ['G1', 'G2'],
                         font_size:12,
                         font_color:'k',
                         font_family:'sans-serif',
                         font_weight:'normal',
                         alpha:None,
                         bbox:None,
                         horizontalalignment:'center',
                         verticalalignment:'center'}}

        note that the name of the keys is not used.
         """
        if gene_list is not None:
            assert len(gene_list) == len(color_scheme_nodes)-1, \
                "ERROR number of gene lists provided must match the color scheme for nodes"

        if (not return_large) and (len(self.nodes) > 500 ):
            raise IOError('The graph contains more than 500 nodes, if you want to plot this specify return_large=True.')

        G = self.getnxGraph()
        if (gene_list is None) and (node_colors is None):
            node_colors = color_scheme_nodes[0]
        elif node_colors is None:
            additional_gl = set.intersection(*[set(i) for i in gene_list])
            if additional_gl:
                gene_list = [set(gl)-additional_gl for gl in gene_list]
                gene_list.append(additional_gl)
                color_scheme_nodes += ("tab:purple",)
            node_colors = []
            for i, gl in enumerate(gene_list):
                node_colors.append([color_scheme_nodes[i+1] if node in gl else "" for node in G.nodes])
            node_colors = list(map(''.join, zip(*node_colors)))
            node_colors = [i if i else color_scheme_nodes[0] for i in node_colors]
            # node_colors = [color_scheme_nodes[1] if node in gene_list else color_scheme_nodes[0] for node in G.nodes]

            assert len(G.nodes) == len(node_colors), "ERROR number of node colors does not match size of graph"

        if all(isinstance(c, (int, float)) for c in node_colors):  # perform rescaling in case of floats for cmap
            node_colors = np.array(node_colors)
            node_colors = (node_colors - np.min(node_colors))/(np.max(node_colors) - np.min(node_colors))

        if edge_df is not None:
            edges = list(G.edges())
            edge_list = [tuple(pair) for pair in edge_df.values]

            edge_color = [color_scheme_edges[1] if edge in edge_list else color_scheme_edges[0] for edge in edges]
            edge_thickness = [2 if edge in edge_list else 1 for edge in edges]

        else:
            edge_color = color_scheme_edges[0]
            edge_thickness = 1.

        plt.figure()
        # TODO: make this prettier
        if title is not None:
            plt.title(title)

        if labels_dict is None:
            nx.draw(G, with_labels=show_labels,
                    node_size=2e2, node_color=node_colors, edge_color=edge_color, width=edge_thickness, cmap=cmap)

        else:
            pos = nx.drawing.spring_layout(G)  # default to spring layout

            nx.draw(G, pos=pos, with_labels=False,
                    node_size=2e2, node_color=node_colors,
                    edge_color=edge_color, width=edge_thickness, cmap=cmap)

            for label_kwds in labels_dict.values():
                nx.draw_networkx_labels(G, pos, **label_kwds)

        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = [20, 15]

        if filename:
            plt.savefig(save_path+filename+'.png')
            plt.close()
        else:
            plt.show()
    
    def getTrainTestPairs_MStree(self,
            train_ratio=0.7,
            train_validation_ratio=0.7,
            neg_pos_ratio=5,
            check_training_set=True,
            random_state=42):
        pass

    def getTrainTestData(self, train_ratio=0.7, neg_pos_ratio=5,
                         train_validation_ratio=None, return_summary=True, random_state=42):
        '''
        :param: train_ratio: The fraction of samples used for training
        :param: neg_pos_ratio: The ratio of negative examples to positive examples
        :param: method: The sampling method used for generating the pairs:
                - ms_tree: uses a minimum spanning tree to find at least one positive pair for each node
                - balanced: draws approximately (neg_pos_ratio * n_positives) negatives for each gene
        :return: positive and negative pairs for both train and test set (4 lists in total)
        '''

        pos_train, neg_train, pos_val, neg_val, pos_test, neg_test, summary_df = self.getTrainTestPairs_MStree(
                                                                                    train_ratio=train_ratio,
                                                                                    train_validation_ratio=train_validation_ratio,
                                                                                    neg_pos_ratio=neg_pos_ratio,
                                                                                    check_training_set=True,
                                                                                    random_state=random_state)
    
        X_train = np.array(pos_train + neg_train)
        X_val = np.array(pos_val + neg_val)
        X_test = np.array(pos_test + neg_test)
    
        Y_train = np.array([1 for _ in range(len(pos_train))] + [0 for _ in range(len(neg_train))])
        Y_val = np.array([1 for _ in range(len(pos_val))] + [0 for _ in range(len(neg_val))])
        Y_test = np.array([1 for _ in range(len(pos_test))] + [0 for _ in range(len(neg_test))])
        
        if return_summary:
            if train_validation_ratio is None:
                return X_train, X_test, Y_train, Y_test, summary_df
            else:
                return X_train, X_val, X_test, Y_train, Y_val, Y_test, summary_df
        else:
            if train_validation_ratio is None:
                return X_train, X_test, Y_train, Y_test
            else:
                return X_train, X_val, X_test, Y_train, Y_val, Y_test


class UndirectedInteractionNetwork(Graph):
    def __init__(self, interaction_df, colnames=None, verbose=True, keeplargestcomponent=False,
                 allow_self_connected=False):
        super().__init__(interaction_df, colnames, verbose=False,  keeplargestcomponent=keeplargestcomponent,
                         allow_self_connected=allow_self_connected)

        self.interactions.values.sort(axis=1)
        self.interactions = self.interactions.drop_duplicates(['Gene_A', 'Gene_B'])

        if verbose:
            print('%d Nodes and %d interactions' % (len(self.nodes), self.interactions.shape[0]))

        self.clf1 = None

    @classmethod
    def createFullyConnectedNetwork(cls, node_names):
        df = pd.DataFrame(np.array([(n1, n2) for i, n1 in enumerate(node_names)
                          for n2 in node_names[:i]]),
                          columns=['gene_A', 'Gene_B'])

        return cls(df)

    def makeSelfConnected(self, inplace=False):
        if not inplace:
            return UndirectedInteractionNetwork(super(UndirectedInteractionNetwork, self).
                                                makeSelfConnected(inplace=False), colnames=('Gene_A', 'Gene_B'),
                                                allow_self_connected=True)

    def checkInteractions_df(self, df, colnames=('Gene_A', 'Gene_B')):
        '''
            Checks which interactions from a given df can be found in the interaction network
        '''
        df.values.sort(axis=1)
        named_net_df = self.getInteractionNamed()
        named_net_df.values.sort(axis=1)
        tester_pairs = set(zip(named_net_df.Gene_A, named_net_df.Gene_B))
        df['In Network'] = [pair in tester_pairs for pair in zip(df[colnames[0]], df[colnames[1]])]
        return df

    def getAdjMatrix(self, sort='first', as_df=False):
        A, nodes = super().getAdjMatrix(sort=sort)

        if as_df:
            return pd.DataFrame(np.maximum(A, np.transpose(A)), columns=nodes, index=nodes)
        else:
            return np.maximum(A, np.transpose(A)), nodes

    def getnxGraph(self, return_names=True):
        '''return a graph instance of the networkx module'''
        if return_names:
            df = self.getInteractionNamed()
        else:
            df = self.interactions

        return nx.from_pandas_edgelist(df, source='Gene_A', target='Gene_B')

    def getAdjDict(self, return_names=True):
        return to_dict_of_lists(self.getnxGraph(return_names=return_names))

    def getMinimmumSpanningTree(self):
        A = self.getnxGraph()
        T = nx.minimum_spanning_tree(A)

        return list(T.edges)

    def getTrainTestPairs_MStree(self, train_ratio=0.7, train_validation_ratio=None, neg_pos_ratio=5,
                                 check_training_set=False, random_state=42):
        '''
        :param: train_ratio: The fraction of samples used for training
        :param: neg_pos_ratio: The ratio of negative examples to positive examples
        :param: assumption: Whether we work in the open world or  closed world assumption
        :return: positive and negative pairs for both train and test set (4 lists in total)
        '''
        np.random.seed(random_state)
        # To get the negatives we first build the adjacency matrix
        df = self.interactions
        df.values.sort(axis=1)

        allpos_pairs = set(zip(df.Gene_A, df.Gene_B)) # this is also calculated in def positives_split, give as additional argument

        pos_train, pos_validation, pos_test = positives_split(df, allpos_pairs, train_ratio, train_validation_ratio)


        N_neg = np.int(neg_pos_ratio * len(allpos_pairs))
        margin = np.int(0.3 * N_neg)

        row_c = np.random.choice(self.N_nodes, N_neg + margin, replace=True)
        col_c = np.random.choice(self.N_nodes, N_neg + margin, replace=True)

        all_pairs = set([tuple(sorted((r_, c_))) for r_, c_ in zip(row_c, col_c) if (c_ != r_)])

        all_neg = np.array(list(all_pairs.difference(allpos_pairs)), dtype=np.uint16)

        if len(all_neg) > N_neg:
            all_neg = all_neg[:N_neg]
        elif len(all_neg) < N_neg:
            print('The ratio of negatives to positives is lower than the asked %f.'
                  '\nReal ratio: %f' % (neg_pos_ratio, len(all_neg) / len(allpos_pairs)))

        train_ids = np.int(len(all_neg) * train_ratio)
        if train_validation_ratio is not None:
            valid_ids = np.int(len(all_neg) * train_validation_ratio)
            neg_train_temp, neg_test = all_neg[:train_ids], all_neg[train_ids:]
            neg_train, neg_validation = neg_train_temp[:valid_ids], neg_train_temp[valid_ids:]
        else:
            neg_train, neg_validation, neg_test = all_neg[:train_ids], np.array([]), all_neg[train_ids:]

        if check_training_set:
            degrees = self.getDegreeDF(return_names=False)
            degrees.index = degrees.Gene.values

            genes, counts = np.unique(all_neg.flatten(), return_counts=True)
            df = pd.DataFrame({'Gene': [self.int2gene[g] for g in genes], 'Counts': counts,
                               'Expected': degrees['Count'].loc[genes].values * neg_pos_ratio})
            df['Difference'] = df.Expected - df.Counts
            return list(pos_train), list(neg_train), list(pos_validation), list(neg_validation), \
                   list(pos_test), list(neg_test), df

        else:
            return list(pos_train), list(neg_train), list(pos_validation), list(neg_validation), \
                   list(pos_test), list(neg_test)



    def getAllTrainData(self, neg_pos_ratio=5, random_state=42):
        np.random.seed(random_state)
        df = self.interactions
        df.values.sort(axis=1)

        X = set(zip(df.Gene_A, df.Gene_B))
        Y_pos = [1 for _ in range(len(X))]

        N_neg = np.int(neg_pos_ratio * len(X))
        all_pairs = set([(i, j) for i in np.arange(self.N_nodes, dtype=np.uint16) for j
                         in np.arange(i + 1, dtype=np.uint16)])

        all_neg = list(all_pairs.difference(X))
        ids = np.random.choice(len(all_neg), np.minimum(N_neg, len(all_neg)), replace=False)

        X = np.array(list(X) + [all_neg[i] for i in ids])
        Y = np.array( Y_pos + [0 for _ in range(len(ids))])

        return X, Y


    def predict_full_matrix(self, sources=None, targets=None, embedding_lookup=None, evaluate=False, Y_true=None,
                            verbose=True, embed_dim=10):
        """
        Uses the fitter from the function 'evaluateEMbeddings'
        :param sources:
        :param targets:
        :param embedding_lookup:
        :param evaluate:
        :param Y_true:
        :param verbose:
        :param embed_dim:
        :return:
        """
        if embedding_lookup is None:
            embedding_lookup = self.embedding_dict

        interactions_to_predict_tuples = product(sources, targets)

        interactions_to_predict = np.array([np.append(embedding_lookup[str(self.gene2int[s])],
                                                      embedding_lookup[str(self.gene2int[t])])
                                            for s, t in interactions_to_predict_tuples])

        assert interactions_to_predict.shape == tuple((len(sources)*len(targets), embed_dim*2)),'ERROR sumtin wong'

        if self.clf1 is not None:
            y_pred_proba = self.clf1.predict_proba(interactions_to_predict)[:, 1]
            y_pred = self.clf1.predict(interactions_to_predict)
            prob_mat = pd.DataFrame(np.reshape(y_pred_proba, (len(sources), len(targets)), order='F'),
                                    columns=targets,
                                    index=sources)

            if evaluate:
                auc_roc = roc_auc_score(Y_true, y_pred_proba)
                auc_pr = average_precision_score(Y_true, y_pred_proba)
                accuracy = accuracy_score(Y_true, y_pred)
                f1 = f1_score(Y_true, y_pred)
                cm = confusion_matrix(Y_true, y_pred)
                if verbose:
                    print('\n' + '#' * 9 + ' Link Prediction Performance ' + '#' * 9)
                    print(f'AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}')
                    print('#' * 50)
                return prob_mat, tuple((auc_roc, auc_pr, accuracy, f1, cm))
            else:
                return prob_mat
        else:
            raise TypeError("No Classifier model definied, run function evaluateEmbeddings first")


    def plot_degree_distribution(self, degreeDf1=None, degreeDf2=None, title=None, legend=False, save_name=None,
                                 s=5, labels=None):

        if degreeDf1 is None:
            degreeDf1 = self.getDegreeDF()

        fig, ax = plt.subplots()
        ax.scatter(np.arange(1, degreeDf1.shape[0] + 1), degreeDf1['Count'], s=s, label=labels[0])
        if degreeDf2 is not None:
            ax.scatter(np.arange(1, degreeDf2.shape[0] + 1), degreeDf2['Count'], s=s, label=labels[1])
        if title is not None:
            plt.title(title)
        else:
            plt.title(f'nodes degreeDf1 {degreeDf1.shape[0]}  |  nodes degreeDf2 {degreeDf2.shape[0]}')
        if legend:
            plt.legend()
        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name)
            plt.close()


def ind2sub(X, nrows):
    X = np.array(X, dtype=np.uint64)
    col_ids, row_ids = np.divmod(X, nrows)

    return row_ids.astype(np.uint16), col_ids.astype(np.uint16)


def sub2ind(X, nrows):
    X = np.array(X)
    ind = X[:, 0] + X[:, 1] * nrows

    return ind


def positives_split(interaction_df, all_pos_pairs=None, train_ratio=0.7, train_validation_ratio=None):
    # First start with the positives
    if all_pos_pairs is None:
        all_pos_pairs = set(zip(interaction_df.Gene_A, interaction_df.Gene_B))
    N_edges = len(all_pos_pairs)
    min_tree = nx.minimum_spanning_tree(nx.from_pandas_edgelist(interaction_df, source='Gene_A', target='Gene_B')).edges

    pos_samples_train = set([tuple(sorted(tup)) for tup in list(min_tree)])
    all_pos_pairs = list(all_pos_pairs.difference(pos_samples_train))

    # determine how much samples need to drawn from the remaining positive pairs to achieve the train_test ratio
    still_needed_samples = np.maximum(0, np.round(train_ratio*N_edges - len(pos_samples_train)).astype(np.int))

    if still_needed_samples == 0:
        print('The train ratio has been increased to include every node in the training set.')

    ids_train = np.random.choice(len(all_pos_pairs), still_needed_samples, replace=False)
    ids_test = np.setdiff1d(np.arange(len(all_pos_pairs)), ids_train)

    pos_samples_train = list(pos_samples_train) + [all_pos_pairs[i] for i in ids_train]
    pos_samples_test = [all_pos_pairs[i] for i in ids_test]

    if train_validation_ratio is not None:
        # pdb.set_trace()
        min_tree_val = nx.minimum_spanning_tree(
            nx.from_pandas_edgelist(pd.DataFrame(pos_samples_train, columns=['Gene_A', 'Gene_B']), source='Gene_A', target='Gene_B')).edges
        pos_samples_train_valid = set([tuple(sorted(tup)) for tup in list(min_tree_val)])
        all_pos_pairs_valid = list(set(pos_samples_train).difference(pos_samples_train_valid))
        N_edges_valid = len(all_pos_pairs_valid)
        still_needed_samples_valid = np.maximum(0, np.round(train_validation_ratio * N_edges_valid - len(pos_samples_train_valid)).astype(np.int))
        ids_train_valid = np.random.choice(len(all_pos_pairs_valid), still_needed_samples_valid, replace=False)
        ids_test_valid = np.setdiff1d(np.arange(len(all_pos_pairs_valid)), ids_train_valid)

        pos_samples_train = list(pos_samples_train_valid) + [all_pos_pairs_valid[i] for i in ids_train_valid]
        pos_samples_valid = [all_pos_pairs_valid[i] for i in ids_test_valid]
    else:
        pos_samples_valid = []

    return pos_samples_train, pos_samples_valid, pos_samples_test


def checkTrainingSetsPairs(X_train, Y_train, X_test, Y_test):
    '''
    :param X_train: a (train_samples, 2) np array
    :param Y_train: a (train_samples, ) np array
    :param X_test: a (test_samples, 2) np array
    :param Y_test: a (test_samples, ) np array
    :return: some statistics on the test and train set
    '''

    # First we check whether every pair in the training and the testing set is unique
    X_train_pairs = set(zip(X_train[:, 0], X_train[:, 1]))
    assert X_train.shape[0] == len(X_train_pairs), 'The training set contains non-unique entries.'
    X_test_pairs = set(zip(X_test[:, 0], X_test[:, 1]))
    assert X_test.shape[0] == len(X_test_pairs), 'The test set contains non-unique entries.'

    # Then we check for data leakage
    assert len(X_train_pairs.intersection(X_test_pairs)) == 0, 'Some gene pairs occur in both training and testing set.'

    # We also check if the ratio of the labels is comparable
    print('Positive-Negtative ratio for the training set: %f' % (sum(Y_train)/len(Y_train)))
    print('Positive-Negtative ratio for the test set: %f' % (sum(Y_test)/len(Y_test)))


## Helper functions


def adj_dict_to_df(adj_dict):

    data = np.array([(k, v) for k, vals in adj_dict.items() for v in vals])
    return pd.DataFrame(data, columns=['Gene_A', 'Gene_B'])


def from_df(self, df, weighted=False, directed=False, weights_col=None):
    self.G = nx.DiGraph()
    src_col = df.columns[0]
    dst_col = df.columns[1]
    if directed:
        def read_weighted(src, dst, w):
            self.G.add_edge(src, dst)
            self.G[src][dst]['weight'] = w

    else:
        def read_weighted(src, dst, w):
            self.G.add_edge(src, dst)
            self.G.add_edge(dst, src)
            self.G[src][dst]['weight'] = w
            self.G[dst][src]['weight'] = w
    
    if weights_col is None:
        weights = [1.0 for row in range(df.shape[0])]
    else:
        try:
            weights = df[weights_col].values.astype(float)
        except:
            raise IOError('The weight column is not known.')

    for src, dst, w in zip(df[src_col].values, df[dst_col].values, weights):
        read_weighted(src, dst, w)

    self.encode_node()


def getSecondOrderNeighbors(adj_dict, adj_dict0=None, incl_first_order=True):
    # slwo
    if adj_dict0 is None:
        adj_dict0 = adj_dict

    if incl_first_order:
        return {k: set([l for v_i in list(v) + [k] for l in adj_dict0[v_i]]) for k, v in adj_dict.items()}
    else:
        return {k: set([l for v_i in v for l in adj_dict0[v_i]]) for k, v in adj_dict.items()}

FITTERMAPPER = {'logistic_classifier': LogisticRegression(solver='lbfgs'),
                'random_forest': RandomForestClassifier(),
                'sgd_classifier': SGDClassifier(loss='log')}