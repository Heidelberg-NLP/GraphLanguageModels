from functools import partial
from argparse import Namespace
from pathlib import Path
import numpy as np
import igraph as ig
from typing import List, Tuple, Dict, Set, Optional, Union, Any
from itertools import chain
from tqdm import tqdm
import json

class Args(Namespace):
    def __init__(
        self, 
        kg: str = "conceptnet", 
        radii: List[int] = [1,2],
        splitsizes_per_relation: Tuple[int,int,int] = (800, 100, 100),
        skip_relations: Set[str] = set(),
    ):
        """
        :param kg: name of the knowledge graph
        :param radii: radius when constructing the subgraphs. e.g. 1 means that only one triplet is in the subgraph
        :param splitsizes_per_relation: number of subgraphs in train, development and test set for each relation
        :param skip_relations: relations that are skipped, i.e. they are not used as labels. 
        """
        super().__init__()
        self.kg = kg
        self.radii = radii
        self.splitsizes_per_relation = splitsizes_per_relation
        self.num_graphs_per_relation = sum(self.splitsizes_per_relation)
        self.skip_relations = skip_relations

        self.in_fn = Path(f"data/knowledgegraph/{self.kg}/graph.pkl")

    def get_out_fn(self, split:str, radius:int) -> Path:
        return (Path(f"data/knowledgegraph/{self.kg}/subgraphs/radius={radius}/{split}_{what}.jsonl") for what in ['graph', 'label'])

def sample_triplets(g:ig.Graph, num_graphs_per_relation:int, relations:List[str], seed:int) -> List[Tuple[str,str]]:
    """
    Sample triplets from the graph
    The sampled triplets have only one connection between two concepts in the undirected graph. This avoids ambiguous labels.
    :param g: the graph
    :param num_graphs_per_relation: number of triplets to sample per relation. In total len(relations) * num_graphs_per_relation triplets are sampled.
    :param relations: list of relations to sample for. Train dev and test set are each balanced for these relations.
    :param seed: random seed
    :return: a list of triplets, where triplets are encoded by a tuple of two node names
    :return: a list of the relations between the nodes in the triplets. This are the labels for the relation prediction task.
    """
    # to avoid ambiguous labels: only consider edges where only one relation connects two concepts in the undirected graph
    def aggregate_relations(relations: List[List[str]]) -> List[str]:
        """Aggregate relations by taking the union of all relations between two concepts"""
        return list(chain(*relations))

    np.random.seed(seed)

    # print('    Converting graph to undirected graph...')
    g_undirected = g.as_undirected(combine_edges=aggregate_relations)
    
    # print('    Get potential edges...')
    relations_list = g_undirected.es['relation']  # of all edges
    potential_indices = [i for i, r in enumerate(relations_list) if len(r) == 1]
    
    relations_list = [relations_list[i][0] for i in potential_indices]  # of all edges that have only one relation

    # print('    Sampling edges for each relation...')
    sampled_edges = []
    failed = []
    good = []
    for r in relations:
        relation_potential_indices = [potential_indices[i] for i, rel in enumerate(relations_list) if rel == r]
        if not len(relation_potential_indices) >= num_graphs_per_relation:
            failed.append((r, len(relation_potential_indices), num_graphs_per_relation))
            continue
        else:
            good.append((r, len(relation_potential_indices), num_graphs_per_relation))
        sampled_indices = np.random.choice(relation_potential_indices, size=num_graphs_per_relation, replace=False)
        sampled_edges += [g_undirected.es[i] for i in sampled_indices]
    print(good)
    assert len(failed) == 0, failed
    del failed

    # print('    Get node names...')
    node_names = [
        (g_undirected.vs.find(e.source)['name'], g_undirected.vs.find(e.target)['name'])
        for e in sampled_edges
    ]
    # print('    Get labels...')
    labels = [e['relation'][0] for e in sampled_edges]
    return node_names, labels

def graph_to_list(graph:ig.Graph, triplet:Optional[Tuple[str,str]]=None, label:Optional[str]=None) -> List[Tuple[str,str,str]]:
    """
    Convert a graph to a list of triplets. 
    :param g: the graph
    :param triplet: if given, the relation of the triplet is replaced with `<mask>`. triplet is a tuple of two node names.
    :param label: if given, the label is used to assert that the relation of the triplet is the same as the label. Otherwise it is not used.
    :return: a list of triplets, where triplets are encoded by a tuple (head, reation, tail)
    """
    g = graph.copy()

    if triplet is not None:
        for e in g.es.select(_within=set((g.vs.find(n).index for n in triplet))):
            assert len(e['relation']) == 1, (e['relation'], triplet)
            if label is not None:
                assert e['relation'][0] == label, (e['relation'], triplet, label)
            e['relation'] = ['<mask>']

    triplets = [
        (g.vs.find(e.source)['name'], rel, g.vs.find(e.target)['name'])
        for e in g.es
        for rel in e['relation']
    ]

    return triplets

def save_subgraphs(subgraphs:List[ig.Graph], labels:List[str], r:int, args:Args, seed:int):
    np.random.seed(seed)
    
    train_indices = []
    dev_indices = []
    test_indices = []

    set_labels = list(set(labels))
    set_labels.sort()  # for reproducibility

    for label in set_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        np.random.shuffle(indices)
        assert len(indices) == args.num_graphs_per_relation
        train_indices += indices[:args.splitsizes_per_relation[0]]
        dev_indices += indices[args.splitsizes_per_relation[0]:args.splitsizes_per_relation[0]+args.splitsizes_per_relation[1]]
        test_indices += indices[args.splitsizes_per_relation[0]+args.splitsizes_per_relation[1]:]
    assert set(train_indices).isdisjoint(set(dev_indices))
    assert set(train_indices).isdisjoint(set(test_indices))
    assert set(dev_indices).isdisjoint(set(test_indices))
    assert len(train_indices) == args.splitsizes_per_relation[0] * len(set(labels))
    assert len(dev_indices) == args.splitsizes_per_relation[1] * len(set(labels))
    assert len(test_indices) == args.splitsizes_per_relation[2] * len(set(labels))

    np.random.shuffle(train_indices)
    np.random.shuffle(dev_indices)
    np.random.shuffle(test_indices)

    for split, indices in zip(['train', 'dev', 'test'], [train_indices, dev_indices, test_indices]):
        tmp_subgraphs = [subgraphs[i] for i in indices]
        tmp_labels = [labels[i] for i in indices]

        out_fn_graph, out_fn_label = args.get_out_fn(split=split, radius=r)
        out_fn_graph.parent.mkdir(parents=True, exist_ok=True)
        with out_fn_graph.open('w') as f:
            for s in tmp_subgraphs:
                f.write(json.dumps(s))
                f.write('\n')
        with out_fn_label.open('w') as f:
            for l in tmp_labels:
                f.write(l)
                f.write('\n')

def main(args):
    print("loading graph")
    g = ig.load(args.in_fn, format='pickle')

    print('getting largest connected component')
    g = g.clusters().giant()

    print(g.summary())
    assert g.is_simple()
    assert g.is_connected()
    assert g.is_directed()

    relations = set(chain(*g.es['relation']))
    assert args.skip_relations.issubset(relations), (args.skip_relations, relations)
    relations = list(relations - args.skip_relations)
    relations.sort()  # for reproducibility

    print(f'samlping triplets for {relations = }')
    triplets, labels = sample_triplets(g, num_graphs_per_relation=args.num_graphs_per_relation, relations=relations, seed=0)
    all_nodes = list(set(chain(*triplets)))

    for r in args.radii:
        print(f"### constructing subgraphs with radius {r} ###")
        print("getting neighborhoods")
        neighborhoods = g.neighborhood(vertices=all_nodes, order=r-1, mode='all')
        neighborhoods = {node: neighborhood for node, neighborhood in zip(all_nodes, neighborhoods)}

        print("constructing subgraphs")
        vertices = [list(set(neighborhoods[source] + neighborhoods[target])) for source, target in triplets]
        subgraphs = [g.induced_subgraph(vertices=vs) for vs in tqdm(vertices)]

        print("convert subgraphs to list")
        assert len(subgraphs) == len(triplets) == len(labels), (len(subgraphs), len(triplets), len(labels))
        subgraphs = [graph_to_list(graph=s, triplet=t, label=l) for s, t, l in tqdm(zip(subgraphs, triplets, labels), total=len(subgraphs))]

        print("saving subgraphs")
        save_subgraphs(subgraphs=subgraphs, labels=labels, r=r, args=args, seed=0)
        print()

if __name__ == "__main__":
    print = partial(print, flush=True)

    skip_relations_1000 = {  # for (800,100,100)
        'conceptnet': set(['/r/RelatedTo', '/r/HasA', '/r/CreatedBy', '/r/ReceivesAction', '/r/LocatedNear', '/r/HasFirstSubevent', '/r/HasLastSubevent', '/r/DefinedAs', '/r/Desires', '/r/SymbolOf', '/r/MadeOf'])
    }

    args = Args(
        kg='conceptnet',
        radii=[1,2,3,4,5],
        splitsizes_per_relation=(800,100,100),
        skip_relations=skip_relations_1000['conceptnet'],
    )

    main(args)

    print("done with main")
