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
import h5py

from models.graph_T5.classifier import GraphT5Classifier, DualGraphT5Classifier
from models.graph_T5.graph_t5 import T5TokenizerFast as T5Tokenizer
from models.graph_T5.wrapper_functions import Graph, graph_to_graphT5, graph_to_set_of_triplets, add_text_to_graph_data, get_embedding, Data
from experiments.encoder.text_guided_relation_prediction.train_LM import str2int, str2bool, str2path, str2criterion, str2logging_level, OpenData, data_to_dataT5, get_data_instances, get_batch, chunker, get_accuracy, run_eval_epoch, reset_params, get_metrics

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
        "--get_dev_scores",
        type=str2bool,
        default=True,
        help="whether to compute dev set scores.",
    )
    parser.add_argument(
        "--save_model_dir",
        type=str2path,
        default=None,
        help="directory was save model to. If None, then the directory is cereated from the other arguments.",
    )
    parser.add_argument(
        "--eval_epochs",
        type=str,
        default="all",
        help="epochs to evaluate on. Can be `all` for all epochs, or `1` for only after one epoch.",
    )
    parser.add_argument(
        "--predict_source",
        type=str2bool,
        default=False,
        help="Whether the source of the relation was predicted during training. If True, then the model is a DualGraphT5Classifier. If False, then the model is a GraphT5Classifier.",
    )
    parser.add_argument(
        "--eval_by_num_seen_instances",
        type=str2bool,
        default=False,
        help="Whether to evaluate by the number of seen instances instead of epochs. This mostly changes the file name of the model to load.",
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
    if args.save_model_dir is None:
        args.save_model_dir = Path(f'trained_models/{"short_train_" if args.eval_by_num_seen_instances else ""}text_guided_relation_prediction/graph_representation={args.graph_representation}-modelsize={args.modelsize}-use_text={args.use_text}-use_graph={args.use_graph}-reset_params={args.reset_params}-predict_source={args.predict_source}-params_to_train={args.params_to_train}-seed={args.seed}-entailed_triplets_only={args.entailed_triplets_only}')
    return args

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

    logging.info('connect to data (will be loaded on the fly)')
    data = OpenData(use_graph=args.use_graph, entailed_triplets_only=args.entailed_triplets_only)

    with data as d:
        num_classes = d.attrs['num_labels']
        num_sources = d.attrs['num_sources']
        source_to_index = json.loads(d.attrs['source_to_index'])
        num_train_instances = len(d['train'])

    if args.eval_by_num_seen_instances:
        fns = list(args.save_model_dir.glob('seen_instances_*'))
        assert len(list(fns)) > 0, f'no model found in {args.save_model_dir}'
        fns = {int(fn.name.split('_')[-1]): fn for fn in fns}
        eval_epochs = list(sorted(fns.keys()))  # not actually the epochs, but the number of seen instances
        eval_epochs = [e for e in eval_epochs if e >= 512]
        # eval_epochs = [e for e in eval_epochs if e > 65536]
    else:
        fns = list(args.save_model_dir.glob('epoch_*'))
        assert len(list(fns)) > 0, f'no model found in {args.save_model_dir}'
        fns = {float(fn.name.split('_')[-1]): fn for fn in fns}
        if args.eval_epochs == 'all':
            eval_epochs = [0] + list(sorted(fns.keys()))
        elif args.eval_epochs in ['1', 1]:
            eval_epochs = [1]
        elif args.eval_epochs in ['log']:
            eval_epochs = torch.unique(torch.logspace(-2, 0, 23).round(decimals=2)).tolist()
            eval_epochs = [0] + [round(e, 2) for e in eval_epochs]
        else:
            raise ValueError(f'unknown eval_epochs {args.eval_epochs}')
    # eval_epochs = [e for e in eval_epochs if e >= 0.12]

    for epoch in eval_epochs:
        logging.info(f'load model for epoch / num_stances {epoch}')  # depending on args.eval_by_num_seen_instances
        if epoch in fns.keys():
            if args.predict_source:
                model = DualGraphT5Classifier.from_pretrained(fns[epoch])
            else:
                model = GraphT5Classifier.from_pretrained(fns[epoch])
        else:
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
        model.to(args.device)
        
        # get eval scores
        if args.get_dev_scores:
            logging.info(f'get dev scores for epoch {epoch}')
            dev_metrics = run_eval_epoch(model=model, data=data, batch_size=args.eval_batch_size, device=args.device, split='val', graph_representation=args.graph_representation, eos_usage=args.eos_usage, use_text=args.use_text, max_seq_len=args.max_seq_len, predict_source=args.predict_source, source_to_index=source_to_index)
            if args.predict_source:
                dev_metrics_sources = dev_metrics[1]
                dev_metrics = dev_metrics[0]
            else:
                dev_metrics_sources = {}
            dev_metrics = {"dev/rel/"+k: v for k,v in dev_metrics.items()}
            dev_metrics_sources = {"dev/source/"+k: v for k,v in dev_metrics_sources.items()}
            dev_metrics = {**dev_metrics, **dev_metrics_sources}
        else:
            dev_metrics = {}

        # get test scores
        logging.info(f'get test scores for epoch {epoch}')
        test_metrics = run_eval_epoch(model=model, data=data, batch_size=args.eval_batch_size, device=args.device, split='test', graph_representation=args.graph_representation, eos_usage=args.eos_usage, use_text=args.use_text, max_seq_len=args.max_seq_len, predict_source=args.predict_source, source_to_index=source_to_index)
        if args.predict_source:
            test_metrics_sources = test_metrics[1]
            test_metrics = test_metrics[0]
        else:
            test_metrics_sources = {}
        test_metrics = {"test/rel/"+k: v for k,v in test_metrics.items()}
        test_metrics_sources = {"test/source/"+k: v for k,v in test_metrics_sources.items()}
        test_metrics = {**test_metrics, **test_metrics_sources}

        if args.eval_by_num_seen_instances:
            num_seen_instances = epoch
            epoch = num_seen_instances / num_train_instances
        else:
            num_seen_instances = epoch * num_train_instances
        wandb_log = {
            "epoch": epoch,
            "num_seen_instances": num_seen_instances,
        }
        wandb_log = {**wandb_log, **dev_metrics, **test_metrics}
        wandb.log(
            wandb_log
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
        project=f"GLM-{'ShortTrain' if args.eval_by_num_seen_instances else ''}DevTest-text_guided_relation_prediction",
        name=name,
        # Track hyperparameters and run metadata
        config=args.__dict__,
        group=f'{name}_eos={args.eos_usage}_reset_params={args.reset_params}_gr={args.graph_representation}_ut={args.use_text}_ug={args.use_graph}',
        tags=['LM']
    )

    main(args)

    logging.info("done with main")
