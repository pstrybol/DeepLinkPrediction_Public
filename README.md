# Setting Up the Environment
**Prerequisit: You must have installed python 3.x. All scripts were tested on Ubuntu 18.04.3 LTS**

1. Install DeepLinkPrdiction package (for use of DLP model and manipulation of interaction networks):
    ```
   cd DeepLinkPrediction
   python setup.py install
   ```
   
2. Clone and Install [OpenNE](https://github.com/thunlp/OpenNE) (For methods GraRep, DeepWalk, node2vec and LINE):
    ```
   cd OpenNE/src
   python setup.py install
   ```

3. Install [forked EvalNE repository](https://github.ugent.be/PSTRYBOL/EvalNE) (for baseline methods and benchmarking framework). Clone the repository, then:
    ```
   cd EvalNE
   python setup.py install 
   ``` 

# Demo of the DLP model
As a demonstration of how the DLP model can be called outside of the EvalNE framework, you will find a script called
`DLP_demo.py` inside the `demo` folder. All necessary data is in this `demo` folder and the script can be run in its
entirety in ~298 seconds on a server (hardware: 48-core CPU and 189GB RAM) and ~336 seconds locally (hardware: 4-core CPU
and 16GB RAM). 

# Workflow for a **single** cancer type
**TODO complete this section**
1. Construct heterogeneous network by integrating the LOF screening data of a certain cancer type (`--disease`)
with a functional interaction scaffold (`--ppi_scaffold`) using `scripts/construct_train_test_edges_separate_args.py`.

2. Construct train/test/validation edges using `construct_train_test_edges_separately.py`.

3. Run EvalNE API's 
    - Run  `EvalNE_General_Performance/EvalNE_General_Performance_API.py` to calculate the general performance 
    (on gene-gene and gene-cell line interactions) of every method.
    -  Run  `EvalNE_CellLine_specific_Performance/EvalNE_CellLine_specific_performance_API.py` to calculate the 
    dependency specific (gene-cell line interactions only) performance of every method.
    - Run `EvalNE_CellLine_specific_total_predictions/EvalNE_CellLine_Specific_totalprediction_API.py` to predict a 
    probability for each possible gene-cell line combination
   
4. Run DLP models with the pretrained embeddings of DeepWalk.
    - Run `DLP_baseline_initializer_Performance/DLP_baseline_initializer_performance_API.py` twice to calculate the 
    general and dependency specific performance, respectively.
    - Run `DLP_baseline_initializer_Target_prediction/DLP_baseline_initializer_targetprediction_API.py` to predict a 
    probability for each possible gene-cell line combination

5. Run `scripts/construct_combined_prediction_df.py` to construct a combined pandas dataframe which averages the 
probabilities of each interaction over the three runs.

These 5 steps are automated in `run_single_cancertype.sh` for a single cancer type. To iterate over several
cancer types use `run_several_cancertypes.sh`


## Useful notes
- The parameter `k-step` of the method GraRep needs to be a multiple of the embedding dimension
- The parameter `order` of the method AROPE needs to be equal to the dimension of the `weights` parameter
- EvalNA API scripts are put in separate directories for the following reason: The version of EvalNE used in this work
does not allow for the evaluation of different test sets on the same trained model. Hence, if we want to predict on a separate
test and and also on a separate prediction set (eg all possible combinations of genes - cell line), 2 separate EvalNE runs
are required. Additionally, EvalNE constructs tmp files on each run making it impossible to run EvalNE on 2 separate 
test sets at the same time impossible, **unless** each API call to EvalNE is sitauted in its own separate directory.

#### TODO
- reactome other cancer types (lung is done)