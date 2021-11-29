#!/bin/bash

for type in 'Bile Duct Cancer' 'Brain Cancer' 'Bladder Cancer' 'Breast Cancer' 'Lung Cancer' 'Prostate Cancer' 'Skin Cancer' 'Pan Cancer';
do
  echo $type
  ./run_single_cancertype.sh -d "$type" -p 'reactome' -r 5 -c 3 -o -1.5 -e -0.5 -t 0.8 -v 0.8 -n 128
done
