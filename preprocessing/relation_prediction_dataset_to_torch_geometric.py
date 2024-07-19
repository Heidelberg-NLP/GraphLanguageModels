from functools import partial
from argparse import Namespace
from transformers import T5EncoderModel, T5TokenizerFast
import torch
from tqdm import tqdm 
from pathlib import Path
import sys

import torch_geometric as gtc
from torch_geometric.data import Data

from experiments.encoder.relation_prediction.train_LM import load_data
from models.graph_T5.wrapper_functions import Graph, _get_str2tok

class Args(Namespace):
    def __init__(
            self,
            kg:str='conceptnet',
            dataset_construction:str='random',
            radius:int=1,
            num_masked:int=0,
            model:str='t5-small',
            device:str='cpu',
        ):
        super().__init__()
        self.kg = kg
        self.dataset_construction = dataset_construction
        self.radius = radius
        self.num_masked = num_masked
        self.model = model
        self.device = device

        out_dir = Path(f"data/knowledgegraph/{kg}/subgraphs_{dataset_construction}/num_neighbors=[1,2,2,2,2]/num_masked={num_masked}/radius={radius}/torch_geometric/encoder={model}")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_fn_with_edge_attribute = Path(out_dir, "with_edge_attribute.pt")
        self.out_fn_no_edge_attribute = Path(out_dir, "no_edge_attribute.pt")


def get_embedding(input, model):
    e = model(input).last_hidden_state
    e = e.mean(dim=1).squeeze(dim=0)
    return e

def get_embeddings(graphs:dict, tokenizer, model, old_embeddings=None):
    if old_embeddings is None:
        old_embeddings = {}
    all_tok_dicts = [
        _get_str2tok(g=graph, tokenizer=tokenizer) for graph_per_split in graphs.values() for graph in graph_per_split
    ]  # list of dicts of {concept: tokenized concept}
    all_tok_dicts = {k: torch.tensor(v, device=args.device).unsqueeze(dim=0) for d in all_tok_dicts for k, v in d.items()}  # merge dicts

    # remove concepts that are already in old_embeddings
    all_tok_dicts = {k: v for k, v in all_tok_dicts.items() if k not in old_embeddings.keys()}

    # todo use batching
    embeddings = {concept: get_embedding(input=input, model=model) for concept, input in tqdm(all_tok_dicts.items())}  # dict of {concept: embedding}
    embeddings = {**old_embeddings, **embeddings}  # merge dicts
    return embeddings

def graph_to_torch_geometric(graph:Graph, label, label_to_index:dict, embeddings:dict):
    concepts = list(graph.concepts)
    
    x = torch.stack([embeddings[concept] for concept in concepts], dim=0)

    concept_to_index = {concept: i for i, concept in enumerate(concepts)}
    edges = graph.g  # list of (concept1, relation, concept2)
    neighboring_concepts = [(c1, c2) for c1, r, c2 in edges if r=='<mask>']
    assert len(neighboring_concepts) == 1
    neighboring_concepts = neighboring_concepts[0]
    is_neighbor = [False] * len(concepts)
    for neighbor in neighboring_concepts:
        is_neighbor[concept_to_index[neighbor]] = True
    is_neighbor = torch.tensor(is_neighbor, dtype=torch.bool)

    edge_index = torch.tensor([[concept_to_index[concept1], concept_to_index[concept2]] for concept1, _, concept2 in edges], dtype=torch.long).t().contiguous()
    
    edge_attr = torch.stack([embeddings[relation] for _, relation, _ in edges], dim=0)

    edge_type = [relation for _, relation, _ in edges]

    y = label_to_index[label]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type, y=y, is_neighbor=is_neighbor)
    return data

def main(args, old_embeddings=None, old_model_str=None, old_model=None):
    if old_embeddings is None:
        old_embeddings = {}
    else:
        assert old_model_str == args.model
    if old_model is not None:
        assert old_model_str == args.model

    print('load data')
    graphs, labels, label_to_index = load_data(kg=args.kg, dataset_construction=args.dataset_construction, radius=args.radius, num_masked=args.num_masked)

    print('load model')
    tokenizer = T5TokenizerFast.from_pretrained(args.model)
    if old_model is None:
        model = T5EncoderModel.from_pretrained(args.model)
        model.to(args.device)
    else:
        model = old_model

    print('get embeddings')
    embeddings = get_embeddings(graphs, tokenizer, model, old_embeddings)

    print('convert to torch_geometric')
    gtc_data = {
        split: [
            graph_to_torch_geometric(graph=graph, label=label, label_to_index=label_to_index, embeddings=embeddings) 
            for graph, label in tqdm(zip(graphs[split], labels[split]), total=len(graphs[split]))
        ]
        for split in graphs.keys()
    }

    print('save with edge attribute')
    print(gtc_data['train'][0])
    torch.save(gtc_data, args.out_fn_with_edge_attribute)

    print('save without edge attribute')
    for graphs in gtc_data.values():
        for graph in graphs:
            del graph.edge_attr
    print(gtc_data['train'][0])
    torch.save(gtc_data, args.out_fn_no_edge_attribute)

    return embeddings, args.model, model  # can be reused for other args

if __name__ == "__main__":
    print = partial(print, flush=True)

    old_embeddings = None
    old_model_str = None
    old_model = None

    for dataset_construction in ['random']:
        for radius in [5,4,3,2,1]:
            for num_masked in [0,1,2,3,4,5]:
                print(f"dataset_construction={dataset_construction}, radius={radius}, num_masked={num_masked}")
                args = Args(
                    kg = 'conceptnet',
                    dataset_construction = dataset_construction,
                    radius = radius,
                    num_masked = num_masked,
                    model = 't5-small',
                    device = 'cpu',
                )
                old_embeddings, old_model_str, old_model = main(args, old_embeddings, old_model_str, old_model)

    print("done with main")