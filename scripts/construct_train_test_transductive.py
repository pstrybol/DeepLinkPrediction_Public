from sklearn.externals.joblib import Parallel, delayed
from subprocess import Popen
import shlex
import pandas as pd
import subprocess
import os


if os.getcwd().endswith("DepMap_DeepLinkPrediction_Benchmark"):
    BASE_PATH = '/'.join(os.getcwd().split('/'))
else:
    BASE_PATH = '/'.join(os.getcwd().split('/')[:-1])

disease = 'Lung Cancer'

transductive_df = pd.read_pickle(f"{BASE_PATH}/transductive_setting/{disease.replace(' ', '_')}/"
                                f"transductive_binned_df.pickle")
bin2gene = transductive_df.groupby('bin_').groups

bin_ = 'bin3'
gene = 'LSM3'


def train_test_transductive(bin2gene, bin_, disease):
    for gene in bin2gene[bin_]:
        print(gene)
        cmd = f"python construct_train_test_edges_separate_args_nonparallel.py --disease '{disease}' "\
              f"--screening 'rnai' --ppi_scaffold 'STRING' --npr_ppi 5 --npr_dep 3 --pos_thresh -1.5 "\
              f"--neg_thresh -0.5 --train_ratio 1.0 --val_ratio 0.8 --transductive --bin '{bin_}' --gene '{gene}'"
        os.system(cmd)
        try:
            subprocess.run([cmd], check=True, shell=True)
        except subprocess.CalledProcessError:
            print('wrongcommand')


backend = 'multiprocessing'
path_func = delayed(train_test_transductive)
Parallel(n_jobs=len(bin2gene), verbose=True, backend=backend)(
path_func(bin2gene, bin_, disease) for bin_ in bin2gene.keys())
