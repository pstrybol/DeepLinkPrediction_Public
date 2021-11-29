#!/bin/bash


helpFunction()
{
	echo ""
	echo "Usage: $0 -d disease -p ppi_scaffold"
	echo -e "\t-d Name of the disease"
	echo -e "\t-p Name of the PPI scaffold"
	echo -e "\t-r negative:positive ratio for PPI"
	echo -e "\t-c negative:positive ratio for Cancer Dependencies"
	echo -e "\t-o Positive dependency threshold"
	echo -e "\t-e Negative dependency threshold"
	echo -e "\t-t Training ratio"
	echo -e "\t-v Validation ratio"
	echo -e "\t-n embedding dimension"
	exit 1 # Exit script after printing help
}

while getopts "d:p:r:c:o:e:t:v:n:" flag;
do
    # shellcheck disable=SC2220
    case "${flag}" in
        d) disease=${OPTARG};;
        p) ppi_scaffold=${OPTARG};;
        r) npr_ppi=${OPTARG};;
        c) npr_dep=${OPTARG};;
        o) pos_thresh=${OPTARG};;
        e) neg_thresh=${OPTARG};;
        t) train_ratio=${OPTARG};;
        v) val_ratio=${OPTARG};;
        n) emb_dim=${OPTARG};;
    esac
done

# Print helpFunction in case parameters are empty
if [ -z "$disease" ] || [ -z "$ppi_scaffold" ] || [ -z "$npr_ppi" ] || [ -z "$npr_dep" ] || [ -z "$pos_thresh" ] || [ -z "$neg_thresh" ] || [ -z "$train_ratio" ] || [ -z "$val_ratio" ] || [ -z "$emb_dim" ]
then
	echo "Some or all of the parameters are empty";
	helpFunction
fi

echo "Disease !: $disease"
echo "PPI Scaffold: $ppi_scaffold"
echo "Negative:Positie Ratio for PPI: $npr_ppi"
echo "Negative:Positie Ratio for DEP: $npr_dep"
echo "Positive Dependency Threshold: $pos_thresh"
echo "Negative Dependency Threshold: $neg_thresh"
echo "Training ratio: $train_ratio"
echo "Validation ratio: $val_ratio"
echo "Embedding dimension: $emb_dim"

# Construct heterogeneous graph and dis_df
echo -e "\n Constructing heterogeneous graph"
python scripts/construct_heterogenous_network_args.py --disease "$disease" --ppi_scaffold "$ppi_scaffold" --pos_thresh $pos_thresh

# Run the train test split
echo -e "\nRunning train/test split"
python scripts/construct_train_test_edges_separate_args.py --disease "$disease" --ppi_scaffold "$ppi_scaffold" --npr_ppi $npr_ppi --npr_dep $npr_dep --pos_thresh $pos_thresh --neg_thresh $neg_thresh --train_ratio $train_ratio --val_ratio $val_ratio

echo -e "\nRunning LP benchmark with EvalNE as python API: GENERAL PEROFRMANCE" # save embs = False
cd EvalNE_General_Performance || exit
python EvalNE_General_Performance_API.py --disease "$disease" --ppi_scaffold "$ppi_scaffold" --npr_ppi $npr_ppi --npr_dep $npr_dep --emb_dim $emb_dim
cd ..

echo -e "\nRunning LP benchmark with EvalNE as python API: CELL LINE PERFORMANCE" # save embs = False
cd EvalNE_CellLine_specific_Performance || exit
python EvalNE_CellLine_specific_performance_API.py --disease "$disease" --ppi_scaffold "$ppi_scaffold" --npr_ppi $npr_ppi --npr_dep $npr_dep --emb_dim $emb_dim
cd ..

#echo -e "\nRunning LP benchmark with EvalNE as python API: CELL LINE enrichment with saving ALL predicitons" # save embs = True
#cd EvalNE_CellLine_specific_total_predictions || exit
#python /EvalNE_CellLine_Specific_totalprediction_API.py.py --ppi_scaffold "$ppi_scaffold" --disease "$disease" --npr_ppi $npr_ppi --npr_dep $npr_dep --emb_dim $emb_dim
#cd ..

echo -e "\nRunning DLP model with pretrained embeddings: GENERAL PERFORMANCE and CELL LINE SPECIFIC PERFORMANCE"
cd DLP_baseline_initializer_Performance || exit
python DLP_baseline_initializer_performance_API.py --baseline_method 'deepwalk-opene' --ppi_scaffold "$ppi_scaffold" --disease "$disease" --npr_ppi $npr_ppi --npr_dep $npr_dep --emb_dim $emb_dim
python DLP_baseline_initializer_performance_API.py --baseline_method 'deepwalk-opene' --ppi_scaffold "$ppi_scaffold" --disease "$disease" --npr_ppi $npr_ppi --npr_dep $npr_dep --emb_dim $emb_dim --general_performance
cd ..

#echo -e "\nRunning DLP model with pretrained embeddings: TARGET PREDICTION"
#cd DLP_baseline_initializer_Target_prediction || exit
#python DLP_baseline_initializer_targetprediction_API.py --baseline_method 'deepwalk-opene' --ppi_scaffold "$ppi_scaffold" --disease "$disease" --npr_ppi $npr_ppi --npr_dep $npr_dep --emb_dim $emb_dim
#cd ..
#
#echo -e "\nConstructing the combined prediction dataframe"
#python scripts/construct_combined_prediction_df.py --disease "$disease" --ppi_scaffold "$ppi_scaffold" --embd_dim $emb_dim --npr_ppi $npr_ppi --npr_dep $npr_dep --train_ratio $train_ratio
