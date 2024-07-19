from functools import partial
from argparse import Namespace
import igraph as ig
from sentence_transformers import SentenceTransformer
from pathlib import Path
import igraph as ig
import json
import torch
import numpy as np
from tqdm import tqdm

class Args(Namespace):
    def __init__(self):
        super().__init__()
        self.kg = 'conceptnet'

        self.kg_in_fn = Path(f'data/knowledgegraph/{self.kg}/graph.pkl')
        self.out_fn = Path(f'data/knowledgegraph/{self.kg}/node_embeddings.json')

        self.model_id = 'all-mpnet-base-v2'  # model id of SBERT
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    # load kg
    g = ig.read(args.kg_in_fn, format="pickle")
    node_names = g.vs['name']
    assert len(node_names) == len(set(node_names))
    del g
    node_names.sort()

    # load SBERT
    model = SentenceTransformer(args.model_id, device=args.device)

    # get node embeddings
    node_embeddings = model.encode(node_names, show_progress_bar=True)
    del model

    out_dict = {node_names[i]: node_embeddings[i].tolist() for i in tqdm(range(len(node_names)))}
    del node_embeddings

    print(f'save node embeddings to {args.out_fn}')
    json.dump(out_dict, open(args.out_fn, 'w'))

if __name__ == "__main__":
    print = partial(print, flush=True)

    args = Args()
    if args.device == 'cpu':
        print('WARNING: using cpu')
        print()

    main(args)

    print("done with main")