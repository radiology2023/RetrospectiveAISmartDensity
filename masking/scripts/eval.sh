#!/bin/bash

# Note: to be run from inside scripts folder.
cd ../src || { echo "Could not change directory, aborting..."; exit 1; }

for i in 1 2 3 4 5
do
    # variables to be used for all
    one_or_multi="multi"
    model_name="csaw-m_${one_or_multi}_hot_final_run_${i}"
    loss_type="${one_or_multi}_hot"
    step=5400
    gpu=7

    test_csv="CSAW-M-review/labels/CSAW-M_test.csv"
    test_folder="CSAW-M-review/preprocessed/test"
    checkpoints_path="CSAW-M-review_checkpoints"
    save_preds_to="CSAW-M-review_preds_2/${model_name}.csv"

    python main.py --evaluate  \
                   --model_name $model_name \
                   --loss_type $loss_type \
                   --step $step \
                   --gpu_id $gpu \
                   --test_csv $test_csv --test_folder $test_folder --checkpoints_path $checkpoints_path \
                   --save_preds_to $save_preds_to
done
