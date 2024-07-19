#!/bin/bash

dataset_construction=random
learning_rate=0.005
hidden_channels=64
activation=ReLU

for seed in 0 # 0 1 2 3 4
do
    for gnn_layer in GCNConv # GCNConv GATConv
    do
        for num_layers in 3 # 2 3 4 5
        do
            for radius in 1 # 1 2 3 4 5
            do
                if [ $radius -eq 4 ]
                then
                    num_maskeds="0 1 2 3 4 5"
                else
                    num_maskeds="0"
                fi
                for num_masked in $num_maskeds
                do
                    echo ""
                    echo running $params_to_train $radius $num_masked $seed $graph_representation
                    python experiments/encoder/relation_prediction/train_GNN.py \
                        --seed $seed \
                        --num_masked $num_masked \
                        --radius $radius \
                        --learning_rate $learning_rate \
                        --dataset_construction $dataset_construction \
                        --gnn_layer $gnn_layer \
                        --num_layers $num_layers \
                        --hidden_channels $hidden_channels \
                        --activation $activation
                    echo done $params_to_train $radius $num_masked $seed $graph_representation
                    echo ""
                done
            done
        done
    done
done

