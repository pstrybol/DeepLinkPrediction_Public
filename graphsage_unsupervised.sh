#!/bin/bash

helpFunction()
{
	echo ""
	echo "Usage: $0 -d disease -p ppi_scaffold"
	echo -e "\t-d Name of the disease"
	echo -e "\t-p Name of the PPI scaffold"
	echo -e "\t-n embedding dimension, for graphsage needs to be half??"
	exit 1 # Exit script after printing help
}

while getopts "d:p:n:" flag;
do
    # shellcheck disable=SC2220
    case "${flag}" in
        d) disease=${OPTARG};;
        p) ppi_scaffold=${OPTARG};;
        n) emb_dim=${OPTARG};;
    esac
done

disease_underscore="${disease// /_}"
echo "Disease : $disease"
echo "Disease underscore: $disease_underscore"
echo "PPI Scaffold: $ppi_scaffold"
echo "Embedding dimension: $emb_dim"


cd scripts || exit
echo -e "Preparing input ...\n"
python prepare_input_graphsage.py --ppi_scaffold "$ppi_scaffold" --disease "$disease" --screening 'rnai'
cd ..

cd GraphSAGE || exit
echo -e "Running GraphSAGE ...\n"
/home/bioit/pstrybol/anaconda3/envs/ghraphsage_v1/bin/python -m graphsage.utils ../GraphSAGE_input/"$ppi_scaffold"/"$disease_underscore"/input-G.json ../GraphSAGE_input/"$ppi_scaffold"/"$disease_underscore"/input-walks.txt
/home/bioit/pstrybol/anaconda3/envs/ghraphsage_v1/bin/python -m graphsage.unsupervised_train --train_prefix ../GraphSAGE_input/"$ppi_scaffold"/"$disease_underscore"/input --model graphsage_seq --max_total_steps 1000 --validate_iter 100 --identity_dim 64 --base_log_dir ../GraphSAGE_logdir/ --dim_1 $emb_dim --dim_2 $emb_dim
cd ..

cd scripts || exit
echo -e "Predicting Test edges ...\n"
python parse_output_graphsage.py --ppi_scaffold "$ppi_scaffold" --disease "$disease" --screening 'rnai'
cd ..