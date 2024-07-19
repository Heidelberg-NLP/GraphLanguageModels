from functools import partial
from argparse import Namespace
from pathlib import Path
from tqdm import tqdm
import json

from models.graph_T5.wrapper_functions import Graph

class Args(Namespace):
    def __init__(self, num_masked:int, radius:int):
        super().__init__()

        self.num_masked = num_masked
        self.radius = radius

        self.in_dir = Path(f"data/knowledgegraph/conceptnet/concept_subgraphs_semantic_label-by-degree/num_neighbors=[2,2,2,2,2]/num_masked=0/radius={self.radius}")
        self.out_dir = Path(f"data/knowledgegraph/conceptnet/concept_subgraphs_semantic_label-by-degree/num_neighbors=[2,2,2,2,2]/num_masked={self.num_masked}/radius={self.radius}")

def load_data(dir:str):
    splits = ['train', 'dev', 'test']
    fn_graphs = [Path(dir, f"{split}_graphs.jsonl") for split in splits]
    fn_labels = [Path(dir, f"{split}_labels.jsonl") for split in splits]
    graphs = {split: [Graph(json.loads(l)) for l in tqdm(fn.open('r'))] for split, fn in zip(splits, fn_graphs)}
    labels = {split: fn.open('r').read() for split, fn in zip(splits, fn_labels)}
    return graphs, labels

def save_data(graphs:dict, labels:dict, out_dir:str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in graphs.keys():
        with Path(out_dir, f'{split}_graphs.jsonl').open('w') as f:
            for graph in graphs[split]:
                f.write(json.dumps(graph))
                f.write('\n')
        with Path(out_dir, f'{split}_labels.jsonl').open('w') as f:
            f.write(labels[split])


def main(args):
    print('load data')
    all_graphs, all_labels = load_data(args.in_dir)

    print('mask data')
    masked_graphs = {split: [] for split in all_graphs.keys()}  # initialize masked graphs
    for split, graphs in all_graphs.items():
        for graph in tqdm(graphs):
            graph.mask_neighbors(args.num_masked)
            masked_graphs[split].append(graph.g)

    print('save data')
    save_data(masked_graphs, all_labels, args.out_dir)

if __name__ == "__main__":
    print = partial(print, flush=True)

    num_masked = 1
    for num_masked in [1,2,3,4,5]:
        for radius in [1,2,3,4,5]:
            print(f"### num_masked={num_masked}, radius={radius} ###")
            args = Args(
                radius=radius,
                num_masked=num_masked,
            )

            main(args)
            print()

    print("done with main")
