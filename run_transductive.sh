#!/bin/bash

cd scripts || exit
python construct_train_test_transductive.py
cd ..

cd EvalNE_transductive || exit
python transductive_setting.py
cd ..
