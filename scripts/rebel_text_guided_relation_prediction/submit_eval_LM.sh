#!/bin/bash

save_preds=True

graph_representation=gGLM # set list gGLM lGLM
reset_params=False # False True (set this to True for gGLM and lGLM to reset the parameters, i.e., to run the Graph Transformer baselines gGT and lGT)
modelsize=t5-small # t5-small t5-base t5-large
eval_by_num_seen_instances=True

use_text=FullyConnected # False FullyConnected
use_graph=True # False True
predict_source=True # False True
entailed_triplets_only=False

get_dev_scores=False # False True
eval_epochs=1 # all log 1

eos_usage=False
init_additional_buckets_from=1e6
device=cuda
logging_level=INFO

eval_batch_size=128

for seed in 0 # 0 1 2 3 4
do
    for params_to_train in all # all head
    do
        echo ""
        echo running $graph_representation $use_text $modelsize $params_to_train $seed
        python experiments/encoder/text_guided_relation_prediction/evaluate_LM.py \
            --seed $seed \
            --params_to_train $params_to_train \
            --graph_representation $graph_representation \
            --reset_params $reset_params \
            --modelsize $modelsize \
            --eval_batch_size $eval_batch_size \
            --eos_usage $eos_usage \
            --init_additional_buckets_from $init_additional_buckets_from \
            --device $device \
            --logging_level $logging_level \
            --use_text $use_text \
            --use_graph $use_graph \
            --get_dev_scores $get_dev_scores \
            --eval_epochs $eval_epochs \
            --predict_source $predict_source \
            --eval_by_num_seen_instances $eval_by_num_seen_instances \
            --entailed_triplets_only $entailed_triplets_only \
            --save_preds $save_preds $1
        echo done $graph_representation $use_text $modelsize $params_to_train $seed
        echo ""
    done
done
