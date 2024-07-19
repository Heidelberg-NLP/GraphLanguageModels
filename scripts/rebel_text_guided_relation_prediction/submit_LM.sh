#!/bin/bash

graph_representation=gGLM # set list gGLM lGLM
reset_params=False # False True (set this to True for gGLM and lGLM to reset the parameters, i.e., to run the Graph Transformer baselines gGT and lGT)
num_epochs=1
modelsize=t5-small # t5-small t5-base t5-large
train_batch_size=8
gradient_accumulation_steps=1

use_text=FullyConnected # False FullyConnected
use_graph=True # False True
predict_source=True # False True
entailed_triplets_only=False # False True

continue_training=False
eos_usage=False
init_additional_buckets_from=1e6
num_evals_per_epoch=2
device=cpu # cuda
logging_level=INFO
run_eval=False
save_model=True


for seed in 0 # 0 1 2 3 4
do
    for params_to_train in all # all head
    do
        if [ $params_to_train == "all" ]
        then
            learning_rate=0.0001
        else
            learning_rate=0.005
        fi
        if [ $save_model == True ]
        then
            save_model_dir=trained_models/short_train_text_guided_relation_prediction/graph_representation=$graph_representation-modelsize=$modelsize-use_text=$use_text-use_graph=$use_graph-reset_params=$reset_params-predict_source=$predict_source-params_to_train=$params_to_train-seed=$seed-entailed_triplets_only=$entailed_triplets_only
        else
            save_model_dir=None
        fi
        echo ""
        echo running $graph_representation $use_text $modelsize $params_to_train $seed
        python experiments/encoder/text_guided_relation_prediction/train_LM.py \
            --seed $seed \
            --params_to_train $params_to_train \
            --learning_rate $learning_rate \
            --graph_representation $graph_representation \
            --reset_params $reset_params \
            --num_epochs $num_epochs \
            --modelsize $modelsize \
            --train_batch_size $train_batch_size \
            --gradient_accumulation_steps $gradient_accumulation_steps \
            --eval_batch_size $train_batch_size \
            --eos_usage $eos_usage \
            --init_additional_buckets_from $init_additional_buckets_from \
            --device $device \
            --logging_level $logging_level \
            --num_evals_per_epoch $num_evals_per_epoch \
            --use_text $use_text \
            --use_graph $use_graph \
            --run_eval $run_eval \
            --save_model_dir $save_model_dir \
            --continue_training $continue_training \
            --predict_source $predict_source \
            --entailed_triplets_only $entailed_triplets_only
        echo done $graph_representation $use_text $modelsize $params_to_train $seed
        echo ""
    done
done
