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

from models.graph_T5.classifier import GraphT5Classifier
from models.graph_T5.graph_t5 import T5TokenizerFast as T5Tokenizer
from models.graph_T5.wrapper_functions import Graph, graph_to_graphT5, graph_to_set_of_triplets, get_embedding, Data

def add_args_shared(parser: ArgumentParser):
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        help="wandb mode. For example `disabled` to disable wandb, which can be useful for debugging.",
    )
    parser.add_argument(
        "--kg",
        type=str,
        default="conceptnet",
        help="name of the knowledge graph",
    )
    parser.add_argument(
        "--dataset_construction",
        type=str,
        default="semantic",
        help="how the dataset is constructed. 'semantic' means that the dataset is constructed by selecting neighbors according to their semantic similarity. 'random' means that the dataset is constructed by sampling neighbors uniformly.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=1,
        help="radius of the subgraphs. e.g. 1 means that only one triplet is in the subgraph",
    )
    parser.add_argument(
        "--num_masked",
        type=int,
        default=0,
        help="size of masked subgraph. 0 means that only the relation to be predicted is masked. 1 means that neighboring concepts are masked as well. 2 means that additionally the next relations are masked. 3 means that additionally the next concepts are masked. etc.",
    )
    parser.add_argument(
        "--modelsize",
        type=str,
        default="t5-small",
        help="size of the model",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="batch size for training",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="number of epochs",
    )
    parser.add_argument(
        '--early_stopping',
        type=int,
        default=5,
        help='number of epochs without improvement before stopping training',
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--optimizer",
        type=str2optimizer,
        default="AdamW",
        help="optimizer",
    )
    parser.add_argument(
        "--criterion",
        type=str2criterion,
        default="CrossEntropyLoss",
        help="criterion, i.e. loss function",
    )
    parser.add_argument(
        "--logging_level",
        type=str2logging_level,
        default="INFO",
        help="logging level",
    )
    parser.add_argument(
        "--wandb_name_prefix",
        type=str,
        default="",
        help="prefix to run name in wandb",
    )

def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--params_to_train",
        type=str,
        default="all",
        help="which parameters to train. 'all' means all parameters. 'head' means only the parameters that are added on top of the pretrained model.",
    )
    parser.add_argument(
        "--graph_representation",
        type=str,
        default="lGLM",
        help="How the graph is represented. 'lGLM' means local graph language model. 'set' means that the graph is represented as a set of triplets (random order) and that the model is a sequence model. 'gGLM' means global GLM, i.e. the same as lGLM but the attention is not sparse and non-neighboring relations and concepts have a PE of the maximum distance. 'list' means that the graph is represented as a list of triplets (alphabetical oder) and that the model is a sequence model.",
    )
    parser.add_argument(
        "--reset_params",
        type=str2bool,
        default=False,
        help="whether to reset the parameters of the model before training. This removes pretrained weights.",
    )
    parser.add_argument(
        "--reload_data",
        type=str2bool,
        default=None,
        help="whether to reload the data in every training epoch. If None, then the default value is chosen depending on the value of graph_representation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient accumulation steps. Effective batch size is `train_batch_size * gradient_accumulation_steps`",
    )
    parser.add_argument(
        "--eos_usage",
        type=str,
        default="False",
        help="Only relevant when using GLM. eos stands for end-of-sequence token. Can be `False` for not using an eos token. When using an eos token, there are two ways to use it: `bidirectional` means that the eos token is connected to every other node in the graph, with a relative position of positive infinity (from node to eos) or negative infinity (from eos to node). `unidirectional` means that the eos token is connected to every node in the graph with a relative position of positive infinity (from node to eos), but not the other way around (i.e. no connection from eos to other node). This means, that nodes do not get messages from the eos token, which perceives locality when using the local GLM"
    )
    parser.add_argument(
        "--num_additional_buckets",
        type=int,
        default=None,
        help="number of additional buckets for relative position embedding. If None, then the default depending on the graph_representation is chosen."
    )
    parser.add_argument(
        "--init_additional_buckets_from",
        type=str2int,
        default=1e6,
        help="Specifies from which bucket of the parent model the additional buckets are initialized. init_additional_buckets_from gives the relative position, and the bucket is the one which corresponds to that relative position. If None, then the additional buckets are initialized randomly as determined by from_pretrained().",
    )

def get_args(parser: ArgumentParser):
    args = parser.parse_args()
    if args.reload_data is None:
        if args.graph_representation in ["set"]:
            args.reload_data = True
        elif args.graph_representation in ["lGLM", "gGLM", "list"]:
            args.reload_data = False
        else:
            raise ValueError(f"unknown graph_representation {args.graph_representation}")

    if args.num_additional_buckets is None:
        if args.graph_representation in ["set", "list","lGLM"]:
            args.num_additional_buckets = 0
        elif args.graph_representation in ["gGLM"]:
            args.num_additional_buckets = 1
        else:
            raise ValueError(f"unknown graph_representation {args.graph_representation}")
    if args.eos_usage != 'False' and args.graph_representation not in ['lGLM', 'gGLM']:
        raise ValueError(f"eos_usage can only be used with lGLM or gGLM, but not with {args.graph_representation}")
    return args

def str2bool(s:str)->bool:
    # input can also be bool
    if s in ['True', 'true', '1', True]:
        return True
    elif s in ['False', 'false', '0', False]:
        return False
    elif s in ['None', None]:
        return None
    else:   
        raise ValueError(f"unknown boolean value {s}")

def str2int(s:str)->Optional[int]:
    if s in ['None', None]:
        return None
    else:
        return int(float(s))

def str2optimizer(s:str)->optim.Optimizer:
    if s == "Adam":
        return optim.Adam
    elif s == "SGD":
        return optim.SGD
    elif s == "AdamW":
        return optim.AdamW
    else:
        raise ValueError(f"unknown optimizer {s}")

def str2criterion(s:str)->nn.Module:
    if s == "CrossEntropyLoss":
        return nn.CrossEntropyLoss
    else:
        raise ValueError(f"unknown criterion {s}")

def freeze_params(s:str, model:GraphT5Classifier) -> None:
    """
    :param s: string that specifies which parameters to train
    :param model: model
    """
    if s == "all":  # all
        pass
    elif s == "head":  # only head
        for param in model.t5model.parameters():
            param.requires_grad = False
    elif s == "PE":  # PE and head
        for param in model.t5model.parameters():  # set all encoder parameters to not trainable
            param.requires_grad = False
        for l in model.t5model.encoder.block:  # set PE to trainable
            if l.layer[0].SelfAttention.has_relative_attention_bias:
                logging.info('setting relative_attention_bias to trainable')
                l.layer[0].SelfAttention.relative_attention_bias.requires_grad = True
                assert l.layer[0].SelfAttention.relative_attention_bias.requires_grad
    else:
        raise ValueError(f"unknown parameters_to_train {s}")

def reset_params(model:GraphT5Classifier) -> None:
    for param in model.parameters():
        nn.init.normal_(param)
    for module in model.t5model.modules():
        model.t5model._init_weights(module)

def str2logging_level(s:str):
    if s == "CRITICAL":
        return logging.CRITICAL
    elif s == "WARNING":
        return logging.WARNING
    elif s == "INFO":
        return logging.INFO
    elif s == "DEBUG":
        return logging.DEBUG
    else:
        raise ValueError(f"unknown logging_level {s}")

def load_data(kg, dataset_construction, radius, num_masked):
    splits = ['train', 'dev', 'test']
    fn_graphs = [Path(f"data/knowledgegraph/{kg}/relation_subgraphs_{dataset_construction}/num_neighbors=[1,2,2,2,2]/num_masked={num_masked}/radius={radius}/{split}_graphs.jsonl") for split in splits]
    fn_labels = [Path(f"data/knowledgegraph/{kg}/relation_subgraphs_{dataset_construction}/num_neighbors=[1,2,2,2,2]/num_masked={num_masked}/radius={radius}/{split}_labels.jsonl") for split in splits]
    fn_label2index = Path(f"data/knowledgegraph/{kg}/relation_subgraphs_{dataset_construction}/num_neighbors=[1,2,2,2,2]/label2index.json")

    graphs = {split: [Graph(json.loads(l)) for l in tqdm(fn.open('r'))] for split, fn in zip(splits, fn_graphs)}

    labels = {split: fn.open('r').readlines() for split, fn in zip(splits, fn_labels)}
    for split in splits:
        labels[split] = [l.strip() for l in labels[split] if l.strip()]

    label_to_index = json.load(fn_label2index.open('r'))

    assert set(labels['train']) == set(labels['dev']) == set(labels['test']), (set(labels['train']), set(labels['dev']), set(labels['test']))

    return graphs, labels, label_to_index

def data_to_dataT5(graph:Graph, tokenizer:T5Tokenizer, label:str, label_to_index:dict, graph_representation:str, eos:str):
    """
    :param graph: graph to convert
    :param tokenizer: tokenizer of model
    :param label: label of the relation
    :param label_to_index: mapping from label to index
    :param graph_representation: how to represent the graph. 
    :param eos: end-of-sequence token. Can be `False` for not using an eos token. When using an eos token, there are two ways to use it: `bidirectional` means that the eos token is connected to every other node in the graph, with a relative position of positive infinity (from node to eos) or negative infinity (from eos to node). `unidirectional` means that the eos token is connected to every node in the graph with a relative position of positive infinity (from node to eos), but not the other way around (i.e. no connection from eos to other node). This means, that nodes do not get messages from the eos token, which preserves locality when using the local GLM
    """
    if graph_representation == 'lGLM':
        data = graph_to_graphT5(graph, tokenizer, how='local', eos=eos)
    elif graph_representation == 'set':
        data = graph_to_set_of_triplets(graph, tokenizer, order='random')
    elif graph_representation == 'gGLM':
        data = graph_to_graphT5(graph, tokenizer, how='global', eos=eos)
    elif graph_representation == 'list':
        data = graph_to_set_of_triplets(graph, tokenizer, order='alphabetical')
    else:
        raise ValueError(f"unknown graph_representation {graph_representation}")
    data.label = torch.tensor(label_to_index[label])
    return data

def get_batch(data_instances:List[Data], pad_token_id:int, device:str):
    """
    can be implemented more efficiently with nested tensors, but they are currently unstable
    """
    max_seq_len = max([data.input_ids.shape[1] for data in data_instances])

    if data_instances[0].relative_position is None:
        assert data_instances[0].sparsity_mask is None
        assert data_instances[0].use_additional_bucket is None
        is_sequence_transformer = True
    else:
        assert data_instances[0].sparsity_mask is not None
        assert data_instances[0].use_additional_bucket is not None
        is_sequence_transformer = False

    # intialize tensors
    input_ids = torch.ones((len(data_instances), max_seq_len), dtype=torch.long, device=device) * pad_token_id
    if not is_sequence_transformer:
        relative_position = torch.zeros((len(data_instances), max_seq_len, max_seq_len), dtype=torch.long, device=device)
        sparsity_mask = torch.zeros((len(data_instances), max_seq_len, max_seq_len), dtype=torch.bool, device=device)
        use_additional_bucket = torch.zeros((len(data_instances), max_seq_len, max_seq_len), dtype=torch.bool, device=device)

    # fill tensors
    for i, data in enumerate(data_instances):
        input_ids[i, :data.input_ids.shape[1]] = data.input_ids
        if not is_sequence_transformer:
            relative_position[i, :data.relative_position.shape[1], :data.relative_position.shape[2]] = data.relative_position
            sparsity_mask[i, :data.sparsity_mask.shape[1], :data.sparsity_mask.shape[2]] = data.sparsity_mask
            use_additional_bucket[i, :data.use_additional_bucket.shape[1], :data.use_additional_bucket.shape[2]] = data.use_additional_bucket

    if is_sequence_transformer:
        relative_position = None # [None] * len(data_instances)
        sparsity_mask = None # [None] * len(data_instances)
        use_additional_bucket = None # [None] * len(data_instances)

    indices = [data.indices for data in data_instances]
    label = torch.tensor([data.label for data in data_instances], device=device)

    return input_ids, relative_position, sparsity_mask, use_additional_bucket, indices, label

def chunker(data_list:List[Data], batch_size:int):
    """
    returns a generator that yields batches of size batch_size
    """
    return (data_list[pos:pos + batch_size] for pos in range(0, len(data_list), batch_size))

def get_accuracy(preds:torch.Tensor, label:torch.Tensor):
    """
    :param preds: shape (batch_size, num_classes)
    :param label: shape (batch_size)
    """
    return (preds.argmax(dim=1) == label).sum().item() / len(label) * 100

def run_eval_epoch(model:GraphT5Classifier, data:List[Data], criterion:nn.Module, batch_size:int, device:str):
    with torch.no_grad():
        losses = []
        accuracies = []
        weights = []

        for data_instances in chunker(data, batch_size):
            # create batch
            logging.debug("get batch")
            input_ids, relative_position, sparsity_mask, use_additional_bucket, indices, label = get_batch(data_instances, pad_token_id=model.tokenizer.pad_token_id, device=device)

            logging.debug("forward")
            logits = model.forward(
                input_ids=input_ids,
                relative_position=relative_position,
                sparsity_mask=sparsity_mask,
                use_additional_bucket=use_additional_bucket,
            )

            logging.debug("get embedding")
            logits = torch.cat([
                get_embedding(sequence_embedding=logits[i], indices=indices[i], concept='<mask>', embedding_aggregation='mean')
                for i in range(len(data_instances))
            ], dim=0)

            logging.debug("get loss and accuracy")
            loss = criterion(logits, label)
            accuracy = get_accuracy(logits, label)

            losses.append(loss.item())
            accuracies.append(accuracy)
            weights.append(len(label)) 
        
        logging.debug("aggregate loss and accuracy")
        loss = np.average(losses, weights=weights)
        accuracy = np.average(accuracies, weights=weights)
    return loss, accuracy

def run_train_epoch(model:GraphT5Classifier, data:List[Data], criterion:nn.Module, optimizer:torch.optim.Optimizer, batch_size:int, gradient_accumulation_steps:int, device:str):
    losses = []
    accuracies = []
    weights = []
    optimizer.zero_grad()

    random.shuffle(data)

    for i, data_instances in tqdm(enumerate(chunker(data, batch_size)), total=len(data)//batch_size):
        # create batch
        input_ids, relative_position, sparsity_mask, use_additional_bucket, indices, label = get_batch(data_instances, pad_token_id=model.tokenizer.pad_token_id, device=device)

        logits = model.forward(
            input_ids=input_ids,
            relative_position=relative_position,
            sparsity_mask=sparsity_mask,
            use_additional_bucket=use_additional_bucket,
        )

        logits = torch.cat([
            get_embedding(sequence_embedding=logits[i], indices=indices[i], concept='<mask>', embedding_aggregation='mean')
            for i in range(len(data_instances))
        ], dim=0)

        loss = criterion(logits, label)
        # loss = logits.sum()

        loss.backward()

        if (i+1) % gradient_accumulation_steps == 0 or (i+1) == len(data)//batch_size:
            optimizer.step()
            optimizer.zero_grad()

        accuracy = get_accuracy(logits, label)
        losses.append(loss.item())
        accuracies.append(accuracy)
        weights.append(len(label)) 

    loss = np.average(losses, weights=weights)
    accuracy = np.average(accuracies, weights=weights)
    return loss, accuracy

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
    graphs, labels, label_to_index = load_data(kg=args.kg, dataset_construction=args.dataset_construction, radius=args.radius, num_masked=args.num_masked)

    logging.info('load T5 encoder')
    num_classes = len(label_to_index)
    
    model = GraphT5Classifier(config=GraphT5Classifier.get_config(num_classes=num_classes, modelsize=args.modelsize, num_additional_buckets=args.num_additional_buckets))
    if args.num_additional_buckets != 0:
        logging.info(f'init relative position bias with {args.num_additional_buckets} additional buckets')
        model.t5model.init_relative_position_bias(modelsize=args.modelsize, init_decoder=False, init_additional_buckets_from=args.init_additional_buckets_from)

    if args.reset_params:
        logging.info('resetting model parameters')
        reset_params(model=model)
    model.to(args.device)

    if not args.reload_data:
        logging.info('convert data to T5 input')
        data = {split: [data_to_dataT5(graph, model.tokenizer, label, label_to_index, args.graph_representation, eos=args.eos_usage) for graph, label in tqdm(zip(graphs[split], labels[split]), total=len(labels[split]))] for split in ['train', 'dev', 'test']}

    # loss and optimizer
    criterion = args.criterion()

    # params_to_train = str2params_to_train(s=args.params_to_train, model=model)
    freeze_params(s=args.params_to_train, model=model)
    # optimizer = args.optimizer(params_to_train, lr=args.learning_rate)
    optimizer = args.optimizer(model.parameters(), lr=args.learning_rate)

    best_epoch = 0
    best_dev_accuracy = 0
    best_dev_loss = float('inf')
    best_test_accuracy = 0
    best_test_loss = float('inf')
    stopped_early = False

    logging.info('train the model')
    for epoch in range(args.num_epochs):
        if args.reload_data:
            logging.info('convert data to T5 input')
            data = {split: [data_to_dataT5(graph, model.tokenizer, label, label_to_index, args.graph_representation, eos=args.eos_usage) for graph, label in tqdm(zip(graphs[split], labels[split]), total=len(labels[split]))] for split in ['train', 'dev', 'test']}
            logging.info('train epoch')
        train_loss, train_accuracy = run_train_epoch(model=model, data=data['train'], criterion=criterion, optimizer=optimizer, batch_size=args.train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, device=args.device)
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
    args = get_args(parser)

    # args.device = 'cuda' if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    
    # logging
    root = logging.getLogger()
    root.setLevel(args.logging_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(args.logging_level)
    formatter = logging.Formatter(f"%(asctime)s [%(levelname)s] %(filename)s, Line %(lineno)d\n%(message)s",datefmt=f"%H:%M:%S",)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # logging.basicConfig(
    #     level=args.logging_level,
    #     # format=f"%(asctime)s [%(levelname)s] %(message)s (Line %(lineno)d in %(filename)s)",
    #     format=f"%(asctime)s [%(levelname)s] %(filename)s, Line %(lineno)d\n%(message)s",
    #     datefmt=f"%H:%M:%S",
    # )

    # wandb
    name = f'{args.wandb_name_prefix}{args.graph_representation:_<4}_{args.params_to_train:_<4}_r={args.radius}_m={args.num_masked}_dsc={args.dataset_construction[0]}_eos={args.eos_usage}_init-additional-buckets-from={args.init_additional_buckets_from}'
    wandb_run = wandb.init(
        mode=args.wandb_mode,
        project="GLM-link_prediction-long_train",
        name=name,
        magic=True,
        # Track hyperparameters and run metadata
        config=args.__dict__,
        group=f'{name}_lr={args.learning_rate}_resetparams={args.reset_params}_modelsize={args.modelsize}_eos={args.eos_usage}',
        tags=['LM']
    )

    main(args)

    logging.info("done with main")
