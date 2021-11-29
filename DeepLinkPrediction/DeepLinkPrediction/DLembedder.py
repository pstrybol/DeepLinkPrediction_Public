import numpy as np
import warnings
from keras.layers import Embedding, Input, Dense, Flatten, Dropout, Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_recall_curve, confusion_matrix, f1_score
from sklearn.metrics import roc_curve
import pandas as pd
import keras
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from time import time
from datetime import datetime
import os


class DLembedder:
    def __init__(self, N_nodes, embed_dim, nodes_per_layer=[None],
                 activations_per_layer=[None], random_state=42,
                 seq_length=2,  dropout=0.2, merge_method='weighted_l1', int2genedict=None):
        '''
        :param N_nodes: The number of genes for which to train embeddings
        :param embed_dim: The embedding dimension
        :param layers_per_model: The number of hidden layers, *including* the output layer. If the model will be used to predict multiple outputs, then
        this value should be a dictionary {output_type1: [n_nodes_hidden_layer1, n_nodes_hidden_layer2],
         output_type2: ...}, for single output models a single iterable containing ints suffices (e.g. [32, 32, 1]
        :param seq_length: The length of a sequence (default is sequences of 2)
        :param dropout: the amount of dropout to apply
        :param merge_method: how to obtain an edge representation from the two gene representations
        '''
        
        np.random.seed(random_state)
        

        # First we check the input variables and convert them to dicts
        nodes_per_layer = checkInputIterable(nodes_per_layer, int, input_name='nodes_per_layer',
                                                   type_name='Integer', new_model_name='Single_Output_Model')

        activations_per_layer = checkInputIterable(activations_per_layer, str, input_name='activations_per_layer',
                                                   type_name='String', new_model_name='Single_Output_Model')

        # Check that the nodes and activation have the same model names and the same number of layers
        compareDictsOfIterables(nodes_per_layer, activations_per_layer, dict1_name='Nodes', dict2_name='Activations')

        # Then we can safely start building the architecture of the model
        embedding_layer = Embedding(input_dim=N_nodes, output_dim=embed_dim, input_length=seq_length, name='Embedder')
        sequence_input = Input(shape=(seq_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        if merge_method.lower() == 'flatten':
            x = Flatten(name='FLAT')(embedded_sequences)
        elif merge_method.lower() == 'average':
            av_layer = Lambda(average_, output_shape=[embed_dim], name='AVG')
            x = av_layer(embedded_sequences)
        elif merge_method.lower() == "hadamard":
            h_layer = Lambda(hadamard, output_shape=[embed_dim], name='HAD')
            x = h_layer(embedded_sequences)
        elif merge_method.lower() == "difference":
            diff_layer = Lambda(diff, output_shape=[embed_dim], name='DIFF')
            x = diff_layer(embedded_sequences)
        elif merge_method.lower() == 'weighted_l1':
            diff_layer = Lambda(weighted_l1, output_shape=[embed_dim], name='WL1')
            x = diff_layer(embedded_sequences)
        elif merge_method.lower() == 'weighted_l2':
            diff_layer = Lambda(weighted_l2, output_shape=[embed_dim], name='WL2')
            x = diff_layer(embedded_sequences)
        else:
            warnings.warn('Edge embedding method (merge_method) not understood, continuing with default [weighted_l1]')
            diff_layer = Lambda(weighted_l1, output_shape=[embed_dim], name='WL1')
            x = diff_layer(embedded_sequences)

        outputs = []

        for modelname, hidden_layers in nodes_per_layer.items():

            try:
                for layer, n_nodes in enumerate(hidden_layers):
                    if layer == (len(hidden_layers) - 1):  # last layer needs to have a name
                        preds = Dense(n_nodes, activation=activations_per_layer[modelname][layer], name=modelname)(x)
                        outputs += [preds]
                    else:
                        x = Dense(n_nodes, activation=activations_per_layer[modelname][layer])(x)
                        x = Dropout(dropout)(x)
            except:
                warnings.warn('Please provide the number of nodes per hidden layer as a dictionary containing'
                              ' arrays of integers.')

            self.model = Model(inputs=sequence_input, outputs=outputs)

        self.random_state = random_state
        self.embedding_dim = embed_dim
        self.seq_length = seq_length
        self.embedding_layer = embedding_layer
        self.merge_method = merge_method
        self.nodes = nodes_per_layer
        self.activations = activations_per_layer
        self.model_names = list(self.activations.keys())
        self.n_models = len(self.model_names)
        self.counter = 0
        self.voc_size = N_nodes
        self.int2gene = int2genedict
        self.history = None

    def saveModel(self, path, model_name=None, extension='.json'):
        config_dict = self.getConfigDict(model_name=model_name)

        config_dict['model_file'] = os.path.join(path, 'weights_' + config_dict['saved_name'] + '.h5')

        config_dict['gene_dict_file'] = None

        if self.int2gene is not None:
            config_dict['gene_dict_file'] = os.path.join(path, 'genedict_' + config_dict['saved_name'] + extension)
            self.saveDict(config_dict['gene_dict_file'])

        config_dict['config_file'] = os.path.join(path, 'config_' + config_dict['saved_name'] + extension)
        self.saveWeights(config_dict['model_file'])

        with open(config_dict['config_file'], 'w') as fp:
            json.dump(config_dict, fp, indent=4)

        return config_dict

    def getConfigDict(self, model_name=None):

        if model_name is None:
            model_name = 'DLP_model'

        stamp, day, time = createStamp(return_day_time=True)

        config = {'model_name': model_name,
                  'day': day,
                  'time': time,
                  'timestamp': stamp,
                  'saved_name': model_name + '_' + stamp,
                  'N_nodes': self.voc_size,
                  'Embedding_dim': self.embedding_dim,
                  'nodes_per_layer': self.nodes,
                  'activations_per_layer': self.activations,
                  'merge_method': self.merge_method,
                  'seq_length': self.seq_length,
                  'random_state': self.random_state
                  }

        return config

    def saveWeights(self, fpath):

        if fpath[-3:] != '.h5':
            fpath = fpath + '.h5'

        self.model.save_weights(fpath)

    def loadWeights(self, fpath):
        if fpath[-3:] != '.h5':
            fpath = fpath + '.h5'

        self.model.load_weights(fpath)

    def saveDict(self, fpath):
        if fpath[-5:] != '.json':
            fpath = fpath + '.json'

        with open(fpath, 'w') as fp:
            json.dump(self.int2gene, fp, indent=4)

    def loadDict(self, fpath):
        if fpath[-5:] != '.json':
            fpath = fpath + '.json'

        with open(fpath, 'r') as fp:
             int2gene = json.load(fp)

        self.int2gene = {int(i): gene for i, gene in int2gene.items()}


    def fit(self, X_train, Y_train, validation_data=None, validation_split=0.2, optimizer='adam',
            loss='binary_crossentropy', return_history=False, verbose=2, batch_size=32,
            n_epochs=10, callbacks="default", predefined_embeddings=None, lossWeights=None,
            metrics='accuracy', allow_nans=True, freeze_embedding=False):

        # Check the training and test data
        Y_train = checkInputDict(Y_train, np.ndarray, self.model_names, inputname='Training labels', allowed_type_name='Numpy array')

        if validation_data is not None:
            validation_labels = checkInputDict(validation_data[1], np.ndarray, self.model_names,
                                                inputname='Test labels', allowed_type_name='Numpy array')

            validation_data = (validation_data[0], validation_labels)
        # Check the loss and the metrics
        loss = checkInputDict(loss, str, self.model_names,
                                                inputname='Losses', allowed_type_name='String')
        metrics = checkInputDict(metrics, str, self.model_names,
                                                inputname='Metrics', allowed_type_name='String')
        if lossWeights is None:
            lossWeights = 1.

        lossWeights = checkInputDict(lossWeights, float, self.model_names,
                                                inputname='Loss weights', allowed_type_name='float')

        if predefined_embeddings is not None:
            self.embedding_layer.set_weights([predefined_embeddings])

        if freeze_embedding:
            print("\n\tEmbedding are frozen!\n")
            self.embedding_layer.trainable = False

        # If the user allows for NaNs we map every loss/metric to a user-friendly version
        if allow_nans:
            try:
                loss = {model: LOSSMAPPER[loss_] for model, loss_ in loss.items()}
            except KeyError:
                raise KeyError('There is not yet an NaN proof version for all provided losses.'
                               ' Implemented NaN metrics are: %s' %', '.join(list(LOSSMAPPER.keys())))

            try:
                metrics = {model: LOSSMAPPER[metric_] for model, metric_ in metrics.items()}
            except KeyError:
                raise KeyError('There is not yet an NaN proof version for all provided losses.'
                               ' Implemented NaN metrics are: %s' %', '.join(list(LOSSMAPPER.keys())))

        # initialize the optimizer and compile the model
        if self.counter == 0:
            print("Compiling the model...")
            print(loss)
            print(metrics)
            self.model.compile(optimizer=optimizer, loss=loss, loss_weights=lossWeights, metrics=metrics)
            self.counter += 1 # To keep the learning rate to its old value

        if str(callbacks).lower() == 'default':
            if validation_data is not None:
                callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')]
            else:
                callbacks = None

        if validation_data is not None:
            validation_split = 0.

        print("Training the model...")
        idx = np.random.permutation(Y_train['Single_Output_Model'].shape[0])
        Y_train['Single_Output_Model'] = Y_train['Single_Output_Model'][idx]
        self.history = self.model.fit(X_train[idx], Y_train, epochs=n_epochs, validation_data=validation_data,
                                 validation_split=validation_split,
                                 callbacks=callbacks, batch_size=batch_size, verbose=verbose)

        if return_history:
            return self.history

    def getEmbeddings(self):
        return np.asarray(self.embedding_layer.get_weights())[0]

    def predict_proba(self, X):
        
        if self.n_models == 1:
            return self.model.predict(X, batch_size=10_000)
        
        else:
            output = self.model.predict(X, batch_size=10_000)
            return {name: output[i] for i, name in enumerate(self.model_names)}
        
    def predict(self, X):
        return np.round(self.model.predict(X)).astype(np.int)

    def predictProbMatrix(self, sources=None, targets=None, model_name=None):
        '''
        Predicts the output as a node list
        :param sources: A list of nodenames
        :param targets: A list of nodenames
        :return: a DF (len(sources) x len(targets) containing the interaction probabilities between the nodes

        '''

        gene2int = self.getGene2Int()

        if sources is None:
            sources_ints = np.arange(self.voc_size)
            sources = [self.int2gene[i] for i in sources_ints]

        else:
            try:
                sources_ints = [gene2int[node] for node in sources if node in gene2int.keys()]

            except KeyError:
                raise KeyError('The provided list needs to contain the strings with the nodenames.')

        if targets is None:
            targets_ints = np.arange(self.voc_size)
            targets = [self.int2gene[i] for i in targets_ints]

        try:
            targets_ints = [gene2int[node] for node in targets if node in gene2int.keys()]

        except KeyError:
            raise KeyError('The provided list needs to contain the strings with the nodenames.')


        input = np.transpose(np.vstack((np.repeat(sources_ints, len(targets_ints)),
                                        np.tile(targets_ints, len(sources_ints)))))
        
        if self.n_models == 1:
            probs = self.predict_proba(input)
        else:

            if model_name not in self.model_names:
                model_names_string = ', '.join(self.model_names)
                raise IOError('The modelname is not understood. \n'
                              'Please specificy the model_name argument using one of the following names:\n'
                              + model_names_string)
            else:
                probs = self.predict_proba(input)[model_name].flatten()
                        
        prob_mat = pd.DataFrame(np.reshape(probs, (len(sources_ints), len(targets_ints)), order='F'),
                                columns=targets,
                                index=sources)
        return prob_mat

    def getScoreMatrix(self, sources=None, targets=None):
        '''
        Predicts the output as a node list
        :param sources: A list of nodenames, the
        :param targets: A list of nodenames, the total score is 1 across each pathway
        :return: a DF (len(sources) x len(targets) containing the interaction probabilities between the nodes
        '''

        prob_mat = self.predictProbMatrix(sources, targets)
        score_mat = prob_mat/np.mean(prob_mat, axis=0)

        return score_mat


    def getModelPerformance(self, X, Y_true, metric='AUC', verbose=False):
        preds = self.predict_proba(X)

        if metric.lower() == 'auc':
            score = roc_auc_score(Y_true, preds)
            fpr, tpr, _ = roc_curve(Y_true, preds)
            print('The %s score of the model is: %f' % (metric, score))
            return score, fpr, tpr
        elif metric.lower() == 'precision':
            score = average_precision_score(Y_true, preds)
            pr, recall, _ = precision_recall_curve(Y_true, preds)
            print('The %s score of the model is: %f' % (metric, score))
            return score, pr, recall
        elif metric.lower() == 'all':
            average_pr = average_precision_score(Y_true, preds)
            auc = roc_auc_score(Y_true, preds)
            acc = accuracy_score(Y_true, (preds>0.5).astype(np.int_))
            cm = confusion_matrix(Y_true, (preds > 0.5).astype(np.int_))
            f1 = f1_score(Y_true, (preds > 0.5).astype(np.int_))
            if verbose:
                print('\n' + '#' * 9 + ' Link Prediction Performance ' + '#' * 9)
                print(f'AUC-ROC: {auc:.3f}, AUC-PR: {average_pr:.3f}, Accuracy: {acc:.3f}, F1-score: {f1:.3f}')
                print('#' * 50)
            return average_pr, auc, acc, cm, f1
        else:
            #pdb.set_trace()
            score = accuracy_score(Y_true, (preds>0.5).astype(np.int_))
            print('The %s score of the model is: %f' % (metric, score))
            return score

    def plot_loss_acc(self, name_fig, time_epochs=None, name_file=None, save_path=None):
        '''
        Plots the (validation) loss and accuracy.
        :param self: DLEmbedder object
        :param name_fig: Title of the figure
        :param time_epochs: List of times per epoch
        :param name_file: Name of the saved plot
        :param save_path: Path where to save the plot
        :return:
        '''
        hist = self.history.history
        x_ticks = range(len(hist['loss']))
        plt.figure(0)
        plt.plot(hist['loss'])
        if hist['val_loss'] is not None:
            plt.plot(hist['val_loss'])
        plt.xlabel('epoch')
        plt.xticks(x_ticks, [i + 1 for i in x_ticks])
        plt.ylabel('loss')
        if hist['val_loss'] is not None:
            plt.legend(['train', 'validation'], loc='upper right')
        else:
            plt.legend(['train'], loc='upper right')

        if time_epochs:
            plt.title(f"{name_fig} | Loss | {np.mean(time_epochs, dtype=int)}s / epoch")
        else:
            plt.title(f"{name_fig} | Loss")

        if name_file:
            plt.savefig(save_path + f"{name_file}_loss.png")
            plt.close()

        plt.figure(1)
        plt.plot(hist['binary_accuracy'])
        if hist['val_loss'] is not None:
            plt.plot(hist['val_binary_accuracy'])

        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.xticks(x_ticks, [i + 1 for i in x_ticks])
        if hist['val_loss'] is not None:
            plt.legend(['train', 'validation'], loc='lower right')
        else:
            plt.legend(['train'], loc='lower right')

        if time_epochs:
            plt.title(f"{name_fig} | Binary Accuracy | {np.mean(time_epochs, dtype=int)}s / epoch")
        else:
            plt.title(f"{name_fig} | Binary Accuracy")

        if name_file:
            plt.savefig(save_path + f"{name_file}_acc.png")
            plt.close()

    def plot_confusion_matrix(self, Y_true, cm, X=None, multiclass=False, normalize=True,
                              title=None, cmap=plt.cm.Blues, name_file=None, save_path=None, verbose=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        if X is not None:
            preds = self.predict_proba(X)
            cm = confusion_matrix(Y_true, (preds > 0.5).astype(np.int_))

        # Only use the labels that appear in the data
        if multiclass:
            classes = unique_labels(Y_true, (preds > 0.5).astype(np.int_))
        else:
            classes = [0, 1]

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        if verbose:
            print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        if name_file:
            plt.savefig(save_path + f"{name_file}_CM.png")
            plt.close()
        return ax, cm


class TimingCallback(keras.callbacks.Callback):
    '''
    Class to store the time of each epoch
    '''
    def on_train_begin(self, logs=None):
        self.logs=[]

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime=time()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time()-self.starttime)


### Helper functions to check inputs

def checkIterableType(iterable, expected_type, input_name='', type_name= ''):
    '''
    Checks whether all elements in a iterable belong to an expected_type (e.g. int, str, ...). returns an error if
    not all elements in the iterable are of expected_type or if the iterable is in fact not an iterable.
    :param iterable: a iterable, returns an error if there can be iterated over this object
    :param expected_type: the expected type for all elements of the iterable
    :param input_name: a string specifying what the iterable represents in the model,
     used to create more informative errors
    :param type_name: A string specifying the expected_type, used to create more informative errors
    '''
    try:
        bools = [isinstance(x, expected_type) for x in iterable]
    except TypeError:
        raise TypeError('Input ' + input_name + ' expected to be an iterable.')

    assert sum(bools) == len(bools), 'Input ' + input_name + ' should consist of all ' + type_name


def checkInputIterable(iterable, expected_type, input_name='', type_name= '', new_model_name='Single_Output_Model'):
    '''
    Checks if an object is a dict or not. If the object is a dict, then all values are verified to be arrays consisting of
    expected type. If the object is another type of iterable, then every element is checked to belong to be of expected_type
    The iterable is converted to a dictionary {new_model_name: iterable}
    :param iterable: a iterable, returns an error if there can be iterated over this object
    :param expected_type: the expected type for all elements of the iterable
    :param input_name: a string specifying what the iterable represents in the model,
     used to create more informative errors
    :param type_name: A string specifying the expected_type, used to create more informative errors
    :param new_model_name:
    :return: a dictionary containing arrays of expected_type
    '''
    if isinstance(iterable, dict):
        for model_, activations_ in iterable.items():
            checkIterableType(activations_, expected_type, input_name=input_name, type_name=type_name)
        return iterable
    else:
        checkIterableType(iterable, expected_type, input_name=input_name, type_name=type_name)
        iterable = {new_model_name: iterable}

        return iterable


def compareDictsOfIterables(dict1, dict2, dict1_name='', dict2_name=''):
    '''
    Compares if two dicts (with iterables as values) have the same keys and whether the arrays are of the same length
    :param dict1: a dictionary to be compared
    :param dict2: a second dictionary to be compared
    :param dict1_name: a string specifying the name of the first dictionary, used for generating more precise errors
    :param dict2_name: a string specifying the name of the first dictionary, used for generating more precise errors
    '''
    try:
        bools = [len(dict1[k]) == len(dict2[k]) for k, v in dict1.items()]
    except KeyError:
        raise KeyError('The modelnames of %s and %s are not consistent.' % (dict1_name, dict2_name))
    except TypeError:
        raise TypeError('All iterables specifying %s and %s should be lists or np.arrays.' % (dict1_name, dict2_name))

    assert sum(bools) == len(bools), 'The number of ' + dict1_name + ' and ' + dict2_name +\
                                     ' are not consistent across all models.'


def checkInputDict(input, allowed_type, modelnames, inputname, allowed_type_name=''):
    '''
    Checks if an input is dict consisting of values of allowed_type and keys equal to modelnames,
    else if the input is a scalar, it is converted to a dict, where all the keys are modelnames,
    all of which map to input.
    :param input: a dict or an object of allowed_type
    :param allowed_type: The allowed type() for an instance of input (when a dict), or input dict (when a scalar)
    :param modelnames: keys of the new dict when input is a scalar
    :param inputname: a string speficying the name of the input, used for more informative errors
    :param allowed_type_name:
    :return: a dictionary, all elements of which belong to allowed_type and keys are given by modelnames
    '''
    if isinstance(input, dict):
        assert set(list(input.keys())) == set(modelnames), \
            'The modelnames of ' + inputname + ' are not consistent with the model'

        bools = [isinstance(v, allowed_type) for model, v in input.items()]

        assert sum(bools) == len(bools), \
            'The types of ' + inputname + ' are not all ' + allowed_type_name

        return input
    elif isinstance(input, allowed_type):
        return {modelname_: input for modelname_ in modelnames}
    else:
        raise IOError('The input ' + inputname + ' should be a dictionary or a ' + allowed_type_name)

# edge embedding functions


def diff(X):
    return X[:, 0] - X[:, 1]


def average_(X):
    return (X[:, 0] + X[:, 1]) / 2.0


def hadamard(X):
    return X[:,0] * X[:,1]

def weighted_l1(X):
    return np.abs(X[:, 1] - X[:, 0])


# Source: EvalNE
def weighted_l2(X):
    return np.power(X[:, 1] - X[:, 0], 2)

# Masking versions of the loss functions from keras


def binary_crossentropy_masked(y_true, y_pred):
    mask = ~tf.is_nan(y_true)
    y_pred = tf.boolean_mask(y_pred, mask)
    y_true = tf.boolean_mask(y_true, mask)
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


LOSSMAPPER = {'binary_crossentropy': binary_crossentropy_masked,
              'binary_crossentropy_masked': binary_crossentropy_masked}



def createStamp(return_day_time=False):
    now = datetime.now()

    stamp = ''.join(e for e in str(now).split('.')[0] if e.isalnum())

    if return_day_time:
        day, time = str(now).split(' ')

        return stamp, day, time

    else:
        return stamp