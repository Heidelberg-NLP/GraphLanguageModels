from functools import partial, wraps
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
import h5py
from sklearn.metrics import f1_score
from time import time

from models.graph_T5.classifier import GraphT5Classifier, DualGraphT5Classifier
from models.graph_T5.graph_t5 import T5TokenizerFast as T5Tokenizer
from models.graph_T5.wrapper_functions import Graph, graph_to_graphT5, graph_to_set_of_triplets, add_text_to_graph_data, get_embedding, Data

def add_args_shared(parser: ArgumentParser):
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        help="wandb mode. For example `disabled` to disable wandb, which can be useful for debugging.",
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
        default=1,
        help="number of epochs",
    )
    parser.add_argument(
        '--early_stopping',
        type=int,
        default=1,
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
        "--num_evals_per_epoch",
        type=int,
        default=100,
        help="number of evaluation on dev and test set per epoch. Has to be at least 1. Metrics on the train set are computed for each sub-epoch independently, so they are computed on different subsets of the train set. Dev and test metrics are always evaluated on the entire dev and test set, respectively. This value does not impact the behavior of early stopping. However, it changes the impact of the random seed, so small variations in performance are to be expected when changing this parameter."
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
    parser.add_argument(
        "--use_text",
        type=str,
        default="FullyConnected",
        help="whether and how to use text as input. Can be `False` for not using text, or `FullyConnected` having a full attention matrix with T2G and G2T attention.",
    )
    parser.add_argument(
        "--use_graph",
        type=str2bool,
        default=True,
        help="Whether to use the graph at all. If False, then only the triplet with the masked relation is used.",
    )
    parser.add_argument(
        "--entailed_triplets_only",
        type=str2bool,
        default=False,
        help="Whether to use only entailed triplets. If True, then the model is trained on (i) entailed triplets and (ii) no-relation triplets only. If False, then the model is trained on all triplets. ",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="maximum sequence length. Sequences longer than this are truncated.",
    )
    parser.add_argument(
        "--run_eval",
        type=str2bool,
        default=False,
        help="whether to run evaluation during training.",
    )
    parser.add_argument(
        "--save_model_dir",
        type=str2path,
        default=None,
        help="directory to save model to. If None, then the model is not saved. Otherwise, it is saved every subepoch (even if run_eval is False).",
    )
    parser.add_argument(
        "--save_at",
        type=str,
        default="log_seen_instances",
        help="when to save the model. Can be `epoch` for saving at the end of each subepoch, or `log_seen_instances` for saving after specific numbers of seen instances or `all` for both.",
    )
    parser.add_argument(
        "--stop_training_after_seen_instances",
        type=str2bool,
        default=True,
        help="whether to stop training after a specific number of seen instances. If True, then the training stops after `stop_training_after_seen_instances` seen instances. If False, then the training stops after `num_epochs` epochs or based on early_stopping.",
    )
    parser.add_argument(
        "--continue_training",
        type=str2bool,
        default=False,
        help="Whether to continue training from the model in save_model_dir. The model weights and training data are used as intended, but second order momentum etc from the optimizer is not initialized correctly.",
    )
    parser.add_argument(
        "--predict_source",
        type=str2bool,
        default=False,
        help="Whether to predict the source of the relation. If True, then the model is a DualGraphT5Classifier and the number of classes is the number of relations and the number of sources (3 for text, graph and no_relation). If False, then the model is a GraphT5Classifier and the number of classes is the number of relations.",
    )
    parser.add_argument(
        "--source_prediction_weight",
        type=float,
        default=0.1,
        help="Weight of the source prediction loss. Only relevant when predict_source is True.",
    )


def get_args(parser: ArgumentParser):
    args = parser.parse_args()

    if args.num_additional_buckets is None:
        args.num_additional_buckets = 0
        if args.graph_representation in ["set", "list","lGLM"]:
            args.num_additional_buckets += 0
        elif args.graph_representation in ["gGLM"]:
            args.num_additional_buckets += 1
        else:
            raise ValueError(f"unknown graph_representation {args.graph_representation}")
        if args.use_text in ['False']:
            args.num_additional_buckets += 0
        elif args.use_text in ['FullyConnected']:
            args.num_additional_buckets += 2
        else:
            raise ValueError(f"unknown use_text {args.use_text}")

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

def str2path(s:str)->Optional[Path]:
    if s in ['None', None]:
        return None
    else:
        return Path(s)

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

class OpenData():
    def __init__(self, use_graph:bool, entailed_triplets_only:bool):
        use_graph_name = '' if use_graph else '-triplet_only'
        entailed_triplets_only_name = '-text_entailed_only' if entailed_triplets_only else ''

        self.fn = f'data/rebel_dataset/rebel{use_graph_name}{entailed_triplets_only_name}.hdf5'

    def __enter__(self):
        self.f = h5py.File(self.fn, 'r')
        return self.f

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()

def data_to_dataT5(graph:Graph, text:str, mask_origin:str, tokenizer:T5Tokenizer, label:int, graph_representation:str, eos:str, use_text:str):
    """
    :param graph: graph to convert
    :param text: text to convert
    :param mask_origin: whether the mask is entailed by text (--> text) or not (--> graph)
    :param tokenizer: tokenizer of model
    :param label: label of the relation
    :param graph_representation: how to represent the graph. 
    :param eos: end-of-sequence token. Can be `False` for not using an eos token. When using an eos token, there are two ways to use it: `bidirectional` means that the eos token is connected to every other node in the graph, with a relative position of positive infinity (from node to eos) or negative infinity (from eos to node). `unidirectional` means that the eos token is connected to every node in the graph with a relative position of positive infinity (from node to eos), but not the other way around (i.e. no connection from eos to other node). This means, that nodes do not get messages from the eos token, which preserves locality when using the local GLM
    :param use_text: whether and how to use text as input. Can be `False` for not using text, `FullyConnected` having a full attention matrix with T2G and G2T attention.
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

    add_text_to_graph_data(data=data, text=text, tokenizer=tokenizer, use_text=use_text)

    data.label = torch.tensor(label)
    data.mask_origin = mask_origin
    return data

def get_data_instances(data:OpenData, split:str, data_indices:List[int], tokenizer:T5Tokenizer, graph_representation:str, eos:str, use_text:str) -> List[Data]:
    """
    :param data: data
    :param split: split of data
    :param data_indices: indices of data instances to get
    :param tokenizer: tokenizer of model
    :param graph_representation: how to represent the graph.
    :param eos: how to handle end of sentence token
    :param use_text: how to handle text
    """

    with data as dat:
        ds = [json.loads(dat[split][i]) for i in data_indices]
    
    try:
        data_instances = [
            data_to_dataT5(graph=Graph(d['triplets']), text=d['text'], mask_origin=d['mask_origin'], tokenizer=tokenizer, label=d['label'], graph_representation=graph_representation, eos=eos, use_text=use_text)
            for d in ds
        ]
    except Exception as e:
        logging.error(f"error when processing {ds}")
        d = ds[0]
        graph = Graph(d['triplets'])
        logging.debug(graph)
        raise e
    return data_instances

def get_batch(data:OpenData, split:str, data_indices:List[int], device:str, tokenizer:T5Tokenizer, graph_representation:str, eos:str, use_text:str, max_seq_len:int, predict_source:bool, source_to_index:Dict[str,int]):
    """
    can be implemented more efficiently with nested tensors, but they are currently unstable
    """
    data_instances = get_data_instances(data=data, split=split, data_indices=data_indices, tokenizer=tokenizer, graph_representation=graph_representation, eos=eos, use_text=use_text)

    current_max_seq_len = max([data.input_ids.shape[1] for data in data_instances])
    max_seq_len = min(max_seq_len, current_max_seq_len)

    if data_instances[0].relative_position is None:
        assert data_instances[0].sparsity_mask is None
        assert data_instances[0].use_additional_bucket is None
        is_sequence_transformer = True
    else:
        assert data_instances[0].sparsity_mask is not None
        assert data_instances[0].use_additional_bucket is not None
        is_sequence_transformer = False

    # intialize tensors
    input_ids = torch.ones((len(data_instances), max_seq_len), dtype=torch.long, device=device) * tokenizer.pad_token_id
    if not is_sequence_transformer:
        relative_position = torch.zeros((len(data_instances), max_seq_len, max_seq_len), dtype=torch.long, device=device)
        sparsity_mask = torch.zeros((len(data_instances), max_seq_len, max_seq_len), dtype=torch.bool, device=device)
        use_additional_bucket = torch.zeros((len(data_instances), max_seq_len, max_seq_len), dtype=torch.bool, device=device)

    # fill tensors
    for i, data in enumerate(data_instances):
        instance_len = min(data.input_ids.shape[1], max_seq_len)
        input_ids[i, :instance_len] = data.input_ids[:, :instance_len]
        if not is_sequence_transformer:
            assert data.input_ids.shape[1] == data.relative_position.shape[1] == data.relative_position.shape[2] == data.sparsity_mask.shape[1] == data.sparsity_mask.shape[2] == data.use_additional_bucket.shape[1] == data.use_additional_bucket.shape[2]
            relative_position[i, :instance_len, :instance_len] = data.relative_position[:, :instance_len, :instance_len]
            sparsity_mask[i, :instance_len, :instance_len] = data.sparsity_mask[:, :instance_len, :instance_len]
            use_additional_bucket[i, :instance_len, :instance_len] = data.use_additional_bucket[:, :instance_len, :instance_len]

    if is_sequence_transformer:
        relative_position = None 
        sparsity_mask = None 
        use_additional_bucket = None 

    indices = [data.indices for data in data_instances]
    label = torch.tensor([data.label for data in data_instances], device=device)
    entailed_by_text = torch.tensor([data.mask_origin == 'text' for data in data_instances], device=device, dtype=torch.bool)

    if predict_source:
        source_label = torch.tensor([source_to_index[data.mask_origin] for data in data_instances], device=device, dtype=torch.long)
    else:
        source_label = None

    return input_ids, relative_position, sparsity_mask, use_additional_bucket, indices, label, entailed_by_text, source_label

def chunker(data_list:list, batch_size:int):
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

def get_preds_and_rank(preds:torch.Tensor, labels:torch.Tensor):
    """
    :param preds: shape (batch_size, num_classes)
    :param label: shape (batch_size)
    :return pred_classes: shape (batch_size). Predicted class for each instance.
    :return ranks: shape (batch_size). Rank of the correct class. 0 is the best rank.
    """
    pred_classes = preds.argmax(dim=1)
    # ranks = preds > preds[labels]

    # get logit for correct class for each instacne
    correct_logit = preds[range(preds.shape[0]), labels]
    # get number of logits that are higher than the correct logit for each instance
    ranks = (preds > correct_logit.unsqueeze(1)).sum(dim=1)
    
    return pred_classes, ranks

def mrr_score(ranks:np.ndarray):
    return np.average(1 / (1 + ranks))

def accuracy_score(pred_classes:np.ndarray, labels:np.ndarray):
    return (pred_classes == labels).mean()

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        total_time = end_time - start_time
        logging.info(f'Function {func.__name__} took {total_time:.2f} seconds')
        return result
    return timeit_wrapper

@timeit
def get_metrics(pred_classes:List[int], ranks:List[int], labels:List[int], entailed_by_texts:List[bool]):
    pred_classes = np.array(pred_classes)
    ranks = np.array(ranks)
    labels = np.array(labels)
    entailed_by_texts = np.array(entailed_by_texts)

    micro_f1 = f1_score(y_true=labels, y_pred=pred_classes, average='micro')
    macro_f1 = f1_score(y_true=labels, y_pred=pred_classes, average='macro')
    mrr = mrr_score(ranks)
    accuracy = accuracy_score(pred_classes, labels)

    text_pred_classes = pred_classes[entailed_by_texts]
    text_ranks = ranks[entailed_by_texts]
    text_labels = labels[entailed_by_texts]
    text_micro_f1 = f1_score(y_true=text_labels, y_pred=text_pred_classes, average='micro')
    text_macro_f1 = f1_score(y_true=text_labels, y_pred=text_pred_classes, average='macro')
    text_mrr = mrr_score(text_ranks)
    text_accuracy = accuracy_score(text_pred_classes, text_labels)

    graph_pred_classes = pred_classes[~entailed_by_texts]
    graph_ranks = ranks[~entailed_by_texts]
    graph_labels = labels[~entailed_by_texts]
    graph_micro_f1 = f1_score(y_true=graph_labels, y_pred=graph_pred_classes, average='micro')
    graph_macro_f1 = f1_score(y_true=graph_labels, y_pred=graph_pred_classes, average='macro')
    graph_mrr = mrr_score(graph_ranks)
    graph_accuracy = accuracy_score(graph_pred_classes, graph_labels)

    out = {
        'all/micro_f1': micro_f1,
        'all/macro_f1': macro_f1,
        'all/mrr': mrr,
        'all/accuracy': accuracy,
        'text/micro_f1': text_micro_f1,
        'text/macro_f1': text_macro_f1,
        'text/mrr': text_mrr,
        'text/accuracy': text_accuracy,
        'graph/micro_f1': graph_micro_f1,
        'graph/macro_f1': graph_macro_f1,
        'graph/mrr': graph_mrr,
        'graph/accuracy': graph_accuracy,
    }
    return out

def run_eval_epoch(model:GraphT5Classifier, data:OpenData, batch_size:int, device:str, split:str, graph_representation:str, eos_usage:str, use_text:str, max_seq_len:int, predict_source:bool, source_to_index:Dict[str,int]):
    with torch.no_grad():
        # losses = []
        labels = []
        pred_classes = []
        ranks = []  # ranks of the correct class. 0 is the highest rank.
        entailed_by_texts = []

        if predict_source:
            label_sources = []
            pred_sources = []
            ranks_sources = []


        with data as d:
            data_indicess = list(range(len(d[split]))) 

        for data_indices in tqdm(chunker(data_indicess, batch_size), total=len(data_indicess)//batch_size):
            # create batch
            logging.debug("get batch")
            input_ids, relative_position, sparsity_mask, use_additional_bucket, indices, label, entailed_by_text, source_label = get_batch(data=data, split=split, data_indices=data_indices, device=device, tokenizer=model.tokenizer, graph_representation=graph_representation, eos=eos_usage, use_text=use_text, max_seq_len=max_seq_len, predict_source=predict_source, source_to_index=source_to_index)

            logging.debug("forward")
            logits = model.forward(
                input_ids=input_ids,
                relative_position=relative_position,
                sparsity_mask=sparsity_mask,
                use_additional_bucket=use_additional_bucket
            )

            if predict_source:
                logits_source = logits[1]
                logits = logits[0]
                

            logits = torch.cat([
                get_embedding(sequence_embedding=logits[i], indices=indices[i], concept='<mask>', embedding_aggregation='mean')
                for i in range(len(data_indices))
            ], dim=0)

            pred_class, rank = get_preds_and_rank(preds=logits, labels=label)
            pred_classes = pred_classes + pred_class.tolist()
            ranks = ranks + rank.tolist()
            labels = labels + label.tolist()
            entailed_by_texts = entailed_by_texts + entailed_by_text.tolist()

            if predict_source:
                logits_source = torch.cat([
                    get_embedding(sequence_embedding=logits_source[i], indices=indices[i], concept='<mask>', embedding_aggregation='mean')
                    for i in range(len(data_indices))
                ], dim=0)
                pred_source, rank_source = get_preds_and_rank(preds=logits_source, labels=source_label)
                pred_sources = pred_sources + pred_source.tolist()
                ranks_sources = ranks_sources + rank_source.tolist()
                label_sources = label_sources + source_label.tolist()

    logging.debug("get metrics")
    metrics = get_metrics(pred_classes=pred_classes, ranks=ranks, labels=labels, entailed_by_texts=entailed_by_texts)
    if predict_source:
        metrics_sources = get_metrics(pred_classes=pred_sources, ranks=ranks_sources, labels=label_sources, entailed_by_texts=entailed_by_texts)
        return metrics, metrics_sources
    return metrics

def run_train_epoch(model:GraphT5Classifier, data:OpenData, data_indicess:List[int], criterion:nn.Module, optimizer:torch.optim.Optimizer, batch_size:int, gradient_accumulation_steps:int, device:str, split:str, graph_representation:str, eos_usage:str, use_text:str, predict_source:bool, source_to_index:Dict[str,int], save_model_after_seen_instances:List[int], save_model_dir:Optional[Path], num_seen_instances:int, stop_training_after_seen_instances:bool):
    losses = []
    accuracies = []
    weights = []
    accuracies_source = []
    optimizer.zero_grad()

    random.shuffle(data_indicess)

    for i, data_indices in tqdm(enumerate(chunker(data_indicess, batch_size)), total=len(data_indicess)//batch_size):
        # create batch
        input_ids, relative_position, sparsity_mask, use_additional_bucket, indices, label, entailed_by_text, source_label = get_batch(data=data, split=split, data_indices=data_indices, device=device, tokenizer=model.tokenizer, graph_representation=graph_representation, eos=eos_usage, use_text=use_text, max_seq_len=args.max_seq_len, predict_source=predict_source, source_to_index=source_to_index)
        num_seen_instances += len(label)
        logits = model.forward(
            input_ids=input_ids,
            relative_position=relative_position,
            sparsity_mask=sparsity_mask,
            use_additional_bucket=use_additional_bucket,
        )
        if predict_source:
            logits_source = logits[1]
            logits = logits[0]

        logits = torch.cat([
            get_embedding(sequence_embedding=logits[i], indices=indices[i], concept='<mask>', embedding_aggregation='mean')
            for i in range(len(data_indices))
        ], dim=0)
        
        loss1 = criterion(logits, label)

        if predict_source:
            logits_source = torch.cat([
                get_embedding(sequence_embedding=logits_source[i], indices=indices[i], concept='<mask>', embedding_aggregation='mean')
                for i in range(len(data_indices))
            ], dim=0)
            loss2 = criterion(logits_source, source_label)

            loss = (1 - args.source_prediction_weight) * loss1 + args.source_prediction_weight * loss2
        else:
            loss = loss1

        loss.backward()

        if (i+1) % gradient_accumulation_steps == 0 or (i+1) == len(data_indicess)//batch_size:
            optimizer.step()
            optimizer.zero_grad()
        else:
            if num_seen_instances in save_model_after_seen_instances:
                logging.warning(f'saving for {num_seen_instances=} actually saw less instances, because gradients were not updated')

        accuracy = get_accuracy(logits, label)
        if predict_source:
            accuracy_source = get_accuracy(logits_source, source_label)
        else:
            accuracy_source = 0
        losses.append(loss.item())
        accuracies.append(accuracy)
        accuracies_source.append(accuracy_source)
        weights.append(len(label)) 
        if num_seen_instances in save_model_after_seen_instances:
            logging.info(f'saving model after {sum(weights)} seen instances')
            if save_model_dir is not None:
                model.save_pretrained(save_model_dir.joinpath(f'seen_instances_{num_seen_instances}'))
                tmp_loss = np.average(losses, weights=weights)
                tmp_accuracy = np.average(accuracies, weights=weights)
                tmp_accuracy_source = np.average(accuracies_source, weights=weights)
                wandb_log = {
                    "train/accuracy": tmp_accuracy, 
                    "train/loss": tmp_loss, 
                    "train/accuracy_source": tmp_accuracy_source,
                    "num_seen_instances": num_seen_instances,
                }
                wandb.log(wandb_log)
            else:
                logging.warning('not saving model')
        if num_seen_instances >= max(save_model_after_seen_instances) and stop_training_after_seen_instances:
            break

    loss = np.average(losses, weights=weights)
    accuracy = np.average(accuracies, weights=weights)
    accuracy_source = np.average(accuracies_source, weights=weights)
    return loss, accuracy, accuracy_source, num_seen_instances

def main(args):
    if args.continue_training:
        assert args.save_model_dir is not None
        assert args.save_model_dir.exists()
    else:
        if args.save_model_dir is not None:
            if args.save_model_dir.exists():
                assert len(list(args.save_model_dir.glob('*'))) == 0, f'{args.save_model_dir} is not empty: {list(args.save_model_dir.glob("*"))}'
            args.save_model_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f'saving model to {args.save_model_dir}')
        else:
            logging.warning('not saving model')

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

    logging.info('connect to data (will be loaded on the fly)')
    data = OpenData(use_graph=args.use_graph, entailed_triplets_only=args.entailed_triplets_only)

    with data as d:
        num_classes = d.attrs['num_labels']
        num_sources = d.attrs['num_sources']
        source_to_index = json.loads(d.attrs['source_to_index'])

    if not args.continue_training:    
        logging.info('load model')
        if args.predict_source:
            model = DualGraphT5Classifier(config=DualGraphT5Classifier.get_config(num_classes1=num_classes, num_classes2=num_sources, modelsize=args.modelsize, num_additional_buckets=args.num_additional_buckets))
        else:
            model = GraphT5Classifier(config=GraphT5Classifier.get_config(num_classes=num_classes, modelsize=args.modelsize, num_additional_buckets=args.num_additional_buckets))
        if args.num_additional_buckets != 0:
            logging.info(f'init relative position bias with {args.num_additional_buckets} additional buckets')
            model.t5model.init_relative_position_bias(modelsize=args.modelsize, init_decoder=False, init_additional_buckets_from=args.init_additional_buckets_from)
        if args.reset_params:
            logging.info('resetting model parameters')
            reset_params(model=model)
        last_epoch = -1
    else:
        logging.info('load model from save_model_dir')
        fns = list(args.save_model_dir.glob('epoch_*'))
        assert len(list(fns)) > 0, f'no model found in {args.save_model_dir}'
        fn = max(fns, key=lambda fn: float(fn.name.split('_')[-1]))
        last_epoch = float(fn.name.split('_')[-1])
        logging.info(f'loading model from {fn}')

        if args.predict_source:
            model = DualGraphT5Classifier.from_pretrained(fn)
        else:
            model = GraphT5Classifier.from_pretrained(fn)
    model.to(args.device)

    # loss and optimizer
    criterion = args.criterion()

    freeze_params(s=args.params_to_train, model=model)
    optimizer = args.optimizer(model.parameters(), lr=args.learning_rate)

    best_epoch = 0
    best_dev_accuracy = 0
    best_dev_loss = float('inf')
    best_test_accuracy = 0
    best_test_loss = float('inf')
    stopped_early = False

    if args.save_at in ['log_seen_instances', 'all']:
        save_model_after_seen_instances = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        assert 512 % (args.train_batch_size * args.gradient_accumulation_steps) == 0, f'{args.train_batch_size}, {args.gradient_accumulation_steps}'
    else:
        save_model_after_seen_instances = []

    logging.info('train the model')
    num_seen_instances = 0
    for train_epoch in range(args.num_epochs):
        with data as d:
            train_indices = list(range(len(d['train'])))
        random.shuffle(train_indices)

        for sub_epoch, sub_train_indices in enumerate(chunker(train_indices, len(train_indices) // args.num_evals_per_epoch), 1):
            epoch = train_epoch + (sub_epoch / args.num_evals_per_epoch)

            if args.continue_training and epoch <= last_epoch:
                    num_seen_instances += len(sub_train_indices)
                    continue

            logging.debug(f'epoch {epoch}')

            train_loss, train_accuracy, train_accuracy_source, num_seen_instances = run_train_epoch(model=model, data=data, data_indicess=sub_train_indices, criterion=criterion, optimizer=optimizer, batch_size=args.train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, device=args.device, split='train', graph_representation=args.graph_representation, eos_usage=args.eos_usage, use_text=args.use_text, predict_source=args.predict_source, source_to_index=source_to_index, save_model_after_seen_instances=save_model_after_seen_instances, save_model_dir=args.save_model_dir, num_seen_instances=num_seen_instances, stop_training_after_seen_instances=args.stop_training_after_seen_instances)
            logging.info(f'train - {epoch = } # {train_loss = :.2f} # {train_accuracy = :.2f} # {train_accuracy_source = :.2f}')

            if args.save_model_dir is not None:
                assert args.num_evals_per_epoch <= 100, 'need to adjust the number of digits in the epoch number in the filename. Might also need to be adjusted for loading the model.'
                model.save_pretrained(args.save_model_dir.joinpath(f'epoch_{epoch:.2f}'))

            if args.run_eval:
                # get dev scores
                dev_metrics = run_eval_epoch(model=model, data=data, batch_size=args.eval_batch_size, device=args.device, split='val', graph_representation=args.graph_representation, eos_usage=args.eos_usage, use_text=args.use_text, max_seq_len=args.max_seq_len, predict_source=args.predict_source, source_to_index=source_to_index)
                if args.predict_source:
                    dev_metrics_sources = dev_metrics[1]
                    dev_metrics = dev_metrics[0]
                logging.info(f'dev   - {epoch = } # {dev_metrics["all/accuracy"] = :.2f}')

                # get test scores
                test_metrics = run_eval_epoch(model=model, data=data, batch_size=args.eval_batch_size, device=args.device, split='test', graph_representation=args.graph_representation, eos_usage=args.eos_usage, use_text=args.use_text, max_seq_len=args.max_seq_len, predict_source=args.predict_source, source_to_index=source_to_index)
                if args.predict_source:
                    test_metrics_sources = test_metrics[1]
                    test_metrics = test_metrics[0]
                    
                logging.info(f'test   - {epoch = } # {test_metrics["all/accuracy"] = :.2f}')

                if dev_metrics['all/accuracy'] > best_dev_accuracy:
                    best_epoch = epoch
                    best_dev_accuracy = dev_accuracy
                    best_test_accuracy = test_accuracy
            else:
                dev_accuracy = float('nan')
                test_accuracy = float('nan')

            wandb_log = {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "stopped_early": float(stopped_early),
                "train/accuracy": train_accuracy, "train/loss": train_loss, "train/accuracy_source": train_accuracy_source,
                "num_seen_instances": num_seen_instances,
            }
            if args.run_eval:
                dev_metrics = {f'dev/{k}': v for k, v in dev_metrics.items()}
                test_metrics = {f'test/{k}': v for k, v in test_metrics.items()}
                wandb_log = {**wandb_log, **dev_metrics, **test_metrics}
                if args.predict_source:
                    dev_metrics_sources = {f'dev/source/{k}': v for k, v in dev_metrics_sources.items()}
                    test_metrics_sources = {f'test/source/{k}': v for k, v in test_metrics_sources.items()}
                    wandb_log = {**wandb_log, **dev_metrics_sources, **test_metrics_sources}
            wandb.log(wandb_log)

            last_epoch = epoch

            if args.run_eval and (epoch - best_epoch >= args.early_stopping):
                logging.info(f'stopped early at epoch {epoch}')
                stopped_early = True
                break
            if num_seen_instances >= max(save_model_after_seen_instances) and args.stop_training_after_seen_instances:
                logging.info(f'stopped early at epoch {epoch}. {num_seen_instances=}. {save_model_after_seen_instances=}')
                stopped_early = True
                break
        if stopped_early:
            break


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter  # makes wandb log the default values
    )
    add_args_shared(parser)
    add_args(parser)
    args = get_args(parser)

    # args.device = 'cuda' if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'

    # logging
    logging.basicConfig(
        level=args.logging_level,
        # format=f"%(asctime)s [%(levelname)s] %(message)s (Line %(lineno)d in %(filename)s)",
        format=f"%(asctime)s [%(levelname)s] %(filename)s, Line %(lineno)d\n%(message)s",
        datefmt=f"%H:%M:%S",
    )

    # wandb
    if args.reset_params:
        assert args.graph_representation in ['lGLM', 'gGLM'], f"reset_params can only be used with lGLM or gGLM, but not with {args.graph_representation}"
        gr_name = args.graph_representation[0] + 'GT'
    else:
        gr_name = args.graph_representation
    name = f'{args.wandb_name_prefix}{gr_name:_<4}_{args.params_to_train:_<4}_{args.modelsize}_t={args.use_text}_g={args.use_graph}_ps={args.predict_source}_eto={args.entailed_triplets_only}'
    wandb_run = wandb.init(
        mode=args.wandb_mode,
        project="GLM-ShortTrain-text_guided_relation_prediction",
        name=name,
        # Track hyperparameters and run metadata
        config=args.__dict__,
        group=f'{name}_lr={args.learning_rate}_resetparams={args.reset_params}_modelsize={args.modelsize}_eos={args.eos_usage}',
        tags=['LM']
    )

    main(args)

    logging.info('done with main')
