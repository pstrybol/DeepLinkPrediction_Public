from DeepLinkPrediction.InteractionNetwork import UndirectedInteractionNetwork
import pytest
import pandas as pd


@pytest.fixture(scope='module')
def dependency_df():
    return pd.read_csv('PID_Breast_Cancer.csv', header=0, index_col=0)

@pytest.fixture()
def heterogenous_nw_obj():
    heterogeneous_network = pd.read_csv(f'PID_dependencies.csv')
    return UndirectedInteractionNetwork(heterogeneous_network)

@pytest.fixture()
def h5py_fp():
    return f"/home/bioit/pstrybol/DepMap_DeepLinkPrediction_Benchmark/LP_train_test_splits/"\
           f"reactome_split_nprPPI5_nprDEP3/Lung_Cancer/"\
           f"complete_predset_with_cl2cl.hdf5"
