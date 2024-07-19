from functools import partial
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import json 
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from typing import List, Tuple, Dict, Union, Optional, Generator
import random
import numpy as np
import sys
import os, sys
import logging
import wandb
import torch_geometric as gtc
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch_geometric.nn as gnn
from itertools import chain

from experiments.encoder.relation_prediction.train_LM import add_args_shared, chunker, get_accuracy

def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--gnn_layer", 
        type=str, 
        required=False, 
        default='GCNConv', 
        help="GNN layer to be used"
    )
    parser.add_argument(
        "--num_layers", 
        type=int, 
        required=False, 
        default=4, 
        help="Number of GNN layers"
    )
    parser.add_argument(
        "--hidden_channels", 
        type=int, 
        required=False, 
        default=64, 
        help="Dimension of intermediate representations, i.e. number of in- and out-channels of intermediate layers"
    )
    parser.add_argument(
        "--activation", 
        type=str, 
        required=False, 
        default='ReLU', 
        help="activation (i.e. non-linearity) that is used after each layer"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        required=False,
        default=0.0,
        help="Dropout probability"
    )

GNN_LAYERS = {
    'RGCNConv': gnn.FastRGCNConv, 
    'RGATConv': gnn.RGATConv,
    'GCNConv': gnn.GCNConv,
    'GATConv': gnn.GATConv,
}

ACTIVATIONS = {
    'ReLU': torch.nn.ReLU(inplace = False),
    'LeakyReLU': torch.nn.LeakyReLU(inplace = False),
}

def get_load_edge_attribute(gnn_layer:str):
    no_need_edge_attribute = ['GCNConv', 'GATConv', 'RGCNConv', 'RGATConv']
    need_edge_attribute = []

    if gnn_layer in no_need_edge_attribute:
        return False
    elif gnn_layer in need_edge_attribute:
        return True
    else:
        raise ValueError(f"Unknown gnn_layer {gnn_layer}")

def parse_args(parser: ArgumentParser):
    args = parser.parse_args()
    args.load_edge_attribute = get_load_edge_attribute(args.gnn_layer)
    args.fn_label_to_index = Path(f'data/knowledgegraph/conceptnet/subgraphs_{args.dataset_construction}/num_neighbors=[1,2,2,2,2]/label2index.json')
    return args

def run_eval_epoch(model:gnn.MessagePassing, data:List[Data], criterion:nn.Module, batch_size:int, device:str):
    with torch.no_grad():
        losses = []
        accuracies = []
        weights = []

        for data_instances in chunker(data, batch_size):
            # create batch
            batch = gtc.data.Batch.from_data_list(data_instances)
            batch = batch.to(device)

            logits = model.forward(
                data=batch,
            )

            loss = criterion(logits, batch.y)
            accuracy = get_accuracy(logits, batch.y)

            losses.append(loss.item())
            accuracies.append(accuracy)
            weights.append(len(batch.y)) 
        
        logging.debug("aggregate loss and accuracy")
        loss = np.average(losses, weights=weights)
        accuracy = np.average(accuracies, weights=weights)
    return loss, accuracy

def run_train_epoch(model:gnn.MessagePassing, data:List[Data], criterion:nn.Module, optimizer:torch.optim.Optimizer, batch_size:int, device:str):
    losses = []
    accuracies = []
    weights = []

    random.shuffle(data)

    for i, data_instances in tqdm(enumerate(chunker(data, batch_size)), total=len(data)//batch_size):
        # logging.info(f"batch {i}")
        optimizer.zero_grad()

        # create batch
        batch = gtc.data.Batch.from_data_list(data_instances)  
        # batch.is_neigbor = list(chain.from_iterable(batch.is_neigbor))
        # print(batch)
        # print(batch.batch)
        # print(batch.is_neigbor)
        batch = batch.to(device)

        logits = model.forward(
            data=batch,
        )

        loss = criterion(logits, batch.y)

        loss.backward()

        optimizer.step()

        accuracy = get_accuracy(logits, batch.y)

        losses.append(loss.item())
        accuracies.append(accuracy)
        weights.append(len(batch.y)) 

    loss = np.average(losses, weights=weights)
    accuracy = np.average(accuracies, weights=weights)
    return loss, accuracy

def load_data(kg:str, dataset_construction:str, radius:str, num_masked:str, modelsize:str, load_edge_attribute:bool):
    dir = Path(f"data/knowledgegraph/{kg}/relation_subgraphs_{dataset_construction}/num_neighbors=[1,2,2,2,2]/num_masked={num_masked}/radius={radius}/torch_geometric/encoder={modelsize}")
    if load_edge_attribute:
        fn = Path(dir, 'with_edge_attribute.pt')
    else:
        fn = Path(dir, 'no_edge_attribute.pt')
    data = torch.load(fn)

    # add self loops and turn to undirected graph
    for data_split in data.values():
        for d in data_split:
            d.edge_index, _ = gtc.utils.add_self_loops(d.edge_index)
            d.edge_index = gtc.utils.to_undirected(d.edge_index)

    return data

class GNN(nn.Module):
    def __init__(
        self, 
        gnn_layer: str, 
        num_layers: int, 
        in_channels: int,
        hidden_channels: int, 
        out_channels: int,
        activation: str,
        dropout: float,
    ) -> None:
        """
        :param gnn_layer: what GNN layer is used
        :param num_layers: number of GNN layers
        :param in_channels: dimension of input, i.e. number on in-channels to first layer
        :param hidden_channels: dimension of intermediate representations, i.e. number of in- and out-channels of intermediate layers
        :param out_channels: dimension of representations after final graph layer, i.e. number of out-channels in final graph layers
        :param activation: activation (i.e. non-linearity) that is used after each layer
        :param dropout: dropout probability (applied after each layer)
        """
        super().__init__()
        self.gnn_layer_str = gnn_layer
        self.gnn_layer = GNN_LAYERS[gnn_layer]
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.activation = ACTIVATIONS[activation]
        self.dropout = dropout
        self.heads = None

        self.gnn_layer_kwargs = {}
        self.gnn_final_layer_kwargs = {}
        # if self.num_relations != None:
        #     self.gnn_layer_kwargs['num_relations'] = self.num_relations
        #     self.gnn_final_layer_kwargs['num_relations'] = self.num_relations
        # if self.heads != None:
        #     self.gnn_layer_kwargs['heads'] = self.heads
        #     self.gnn_final_layer_kwargs['heads'] = 1

        # create layers
        assert self.num_layers >= 2, f'Current implementation does not support 1 layer.\n{self.num_layers = }'

        self.first_layer = self.gnn_layer(
            in_channels = self.in_channels,
            out_channels = self.hidden_channels,
            **self.gnn_layer_kwargs
        )

        self.intermediate_layers = torch.nn.ModuleList(
            [
                self.gnn_layer(
                    in_channels = self.hidden_channels * (self.heads if self.heads != None else 1),
                    out_channels = self.hidden_channels,
                    **self.gnn_layer_kwargs
                )
                for _ in range(self.num_layers - 2)
            ]
        )

        self.final_layer = self.gnn_layer(
            in_channels = self.hidden_channels * (self.heads if self.heads != None else 1),
            out_channels = self.out_channels,
            **self.gnn_final_layer_kwargs
        )

        if dropout > 0:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = torch.nn.Identity()

    def _one_layer(
        self, 
        layer: gnn.MessagePassing,
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Performs one layer of the GNN
        :param layer: layer to be applied
        :param x: node features for torch_geometric
        :param edge_index: edge_index for torch_geometric 
        :param edge_type: edge type for RGCN
        :return: node features after layer application
        """
        if self.gnn_layer_str in ['GCNConv', 'GATConv']:
            return layer(x, edge_index)
        
        if self.gnn_layer_str in ['RGCNConv', 'RGATConv']:
            return layer(x, edge_index, edge_type)
        
        raise NotImplementedError(self.gnn_layer_str)

    def forward(
        self, 
        data: Data
    ) -> torch.Tensor:
        """
        :param data: gtc data instace with x, edge_index, is_neighbor and optionally more
        """
        
        x = self.dropout(data.x)
        edge_index = data.edge_index
        edge_type = data.edge_type if hasattr(data, 'edge_type') else None

        x = self._one_layer(self.first_layer, x, edge_index, edge_type)
        x = self.activation(x)

        for l in self.intermediate_layers:
            x = self._one_layer(l, x, edge_index, edge_type)
            x = self.activation(x)

        x = self._one_layer(self.final_layer, x, edge_index, edge_type)

        # mask out all non-neighbors (i.e. nodes with radius > 1)
        assert all(gnn.global_add_pool(data.is_neighbor.unsqueeze(-1)*1.0, data.batch) == 2), (gnn.global_add_pool(data.is_neighbor.unsqueeze(-1)*1.0, data.batch), data.is_neighbor, data.batch)

        x = x*data.is_neighbor.unsqueeze(-1)
        # take batch-wise mean
        x = gnn.global_add_pool(x, data.batch)
        x /= 2

        return x

def main(args):
    if not args.device.startswith('cuda'):
        logging.warning(f'using CPU {args.device}, training might be slow.')
    else:
        logging.info(f'using GPU {args.device}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    logging.info('load data')
    data = load_data(kg=args.kg, dataset_construction=args.dataset_construction, radius=args.radius, num_masked=args.num_masked, modelsize=args.modelsize, load_edge_attribute=args.load_edge_attribute)
    label_to_index = json.load(args.fn_label_to_index.open('r'))

    logging.info('load GNN')
    num_classes = len(label_to_index)
    model = GNN(
        gnn_layer=args.gnn_layer,
        num_layers=args.num_layers,
        in_channels=data['train'][0].x.shape[1],
        hidden_channels=args.hidden_channels, 
        out_channels=num_classes,
        activation=args.activation,
        dropout=args.dropout,
    )
    model = model.to(args.device)

    # loss and optimizer
    criterion = args.criterion()

    optimizer = args.optimizer(model.parameters(), lr=args.learning_rate)

    best_epoch = 0
    best_dev_accuracy = 0
    best_dev_loss = float('inf')
    best_test_accuracy = 0
    best_test_loss = float('inf')
    stopped_early = False

    logging.info('train the model')

    # epoch = 0
    # batch_size = args.train_batch_size
    # device=args.device

    for epoch in range(args.num_epochs):
        train_loss, train_accuracy = run_train_epoch(model=model, data=data['train'], criterion=criterion, optimizer=optimizer, batch_size=args.train_batch_size, device=args.device)
        logging.info(f'train - {epoch = } # {train_loss = :.2f} # {train_accuracy = :.2f}')

        # get dev scores
        dev_loss, dev_accuracy = run_eval_epoch(model=model, data=data['dev'], criterion=criterion, batch_size=args.eval_batch_size, device=args.device)
        logging.info(f'dev   - {epoch = } # {dev_loss = :.2f} # {dev_accuracy = :.2f}')

        # get test scores
        test_loss, test_accuracy = run_eval_epoch(model=model, data=data['test'], criterion=criterion, batch_size=args.eval_batch_size, device=args.device)
        logging.info(f'test  - {epoch = } # {test_loss = :.2f} # {test_accuracy = :.2f}')

        if dev_loss < best_dev_loss:
            best_epoch = epoch
            best_dev_accuracy = dev_accuracy
            best_dev_loss = dev_loss
            best_test_accuracy = test_accuracy
            best_test_loss = test_loss

        wandb.log(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "stopped_early": float(stopped_early),
                "train/accuracy": train_accuracy, "train/loss": train_loss, 
                "dev/accuracy": dev_accuracy, "dev/loss": dev_loss, 'dev/best_accuracy': best_dev_accuracy, 'dev/best_loss': best_dev_loss,
                "test/accuracy": test_accuracy, "test/loss": test_loss, 'test/best_accuracy': best_test_accuracy, 'test/best_loss': best_test_loss,
            }
        )

        last_epoch = epoch
        if epoch - best_epoch >= args.early_stopping:
            logging.info(f'stopped early at epoch {epoch}')
            stopped_early = True
            break

    for epoch in range(last_epoch+1, args.num_epochs):
        wandb.log(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "stopped_early": float(stopped_early),
                "train/accuracy": train_accuracy, "train/loss": train_loss, 
                "dev/accuracy": dev_accuracy, "dev/loss": dev_loss, 'dev/best_accuracy': best_dev_accuracy, 'dev/best_loss': best_dev_loss,
                "test/accuracy": test_accuracy, "test/loss": test_loss, 'test/best_accuracy': best_test_accuracy, 'test/best_loss': best_test_loss,
            }
        )


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter  # makes wandb log the default values
    )
    add_args_shared(parser)
    add_args(parser)
    args = parse_args(parser)

    # args.device = 'cuda' if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    name = f'{args.wandb_name_prefix}GNN={args.gnn_layer:_<4}_r={args.radius}_m={args.num_masked}_dsc={args.dataset_construction[0]}'
    wandb_run = wandb.init(
        mode=args.wandb_mode,
        project="GLM-link_prediction-long_train",
        name=name,
        magic=True,
        # Track hyperparameters and run metadata
        config=args.__dict__,
        group=f'{name}_lr={args.learning_rate}_nl={args.num_layers}_hs={args.hidden_channels}_act={args.activation}_dp={args.dropout}_bs={args.train_batch_size}',
        tags=['GNN'],
    )

    logging.basicConfig(
        level=args.logging_level,
        # format=f"%(asctime)s [%(levelname)s] %(message)s (Line %(lineno)d in %(filename)s)",
        format=f"%(asctime)s [%(levelname)s] %(filename)s, Line %(lineno)d\n%(message)s",
        datefmt=f"%H:%M:%S",
    )

    main(args)

    logging.info("done with main")
