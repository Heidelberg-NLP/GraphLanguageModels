from functools import partial
from argparse import Namespace
import logging
from pathlib import Path
import json
from tqdm import tqdm
from collections import Counter
import pandas as pd
from SPARQLWrapper import SPARQLWrapper
from time import sleep
import random
from datasets import load_dataset
import h5py
import os

class Args(Namespace):
    def __init__(self):
        super().__init__()
        self.add_no_relation = True
        self.no_relation_prob = 0.1
        self.no_relation_label = 'has no relation to'  # this verbalization of the label is not used in the experiments, because it will always be replaced by a mask token
        
        self.in_fns = {
            split: Path(f'data/rebel_dataset/en_{split}.jsonl')
            for split in ['train', 'val', 'test']
            # for split in ['dummy']
        }
        self.label_to_index_fn = Path('data/rebel_dataset/label2index.json')
        self.index_to_label_fn = Path('data/rebel_dataset/index2label.json')
        self.all_relations_fn = Path('data/rebel_dataset/all_relations.json')
        self.wikidata_triplets_fn = Path('data/knowledgegraph/wikidata/wikidata_triples_cleaned.csv')
        self.fns_augmented = {
            split: Path(fn.parent.joinpath(fn.stem + '_augmented.jsonl'))
            for split, fn in self.in_fns.items()
        }
        self.fn_property_surfaceform = Path('data/rebel_dataset/property_surfaceform.tsv')
        self.fns_property_surfaceform = {
            split: Path(fn.parent.joinpath(fn.stem + '_augmented_property_surfaceform.jsonl'))
            for split, fn in self.in_fns.items()
        }
        self.fns_hf_dataset = {
            split: Path(fn.parent.joinpath(fn.stem + '_hf_dataset.jsonl'))
            for split, fn in self.in_fns.items()
        }
        self.fn_hdf5 = 'data/rebel_dataset/rebel.hdf5'
        self.fn_hdf5_tripletonly = 'data/rebel_dataset/rebel-triplet_only.hdf5'
        self.fn_hdf5_textentailedonly = 'data/rebel_dataset/rebel-text_entailed_only.hdf5'
        self.fn_hdf5_tripletonly_textentailedonly = 'data/rebel_dataset/rebel-triplet_only-text_entailed_only.hdf5'

        self.logging_level = logging.DEBUG



def load_jsonl(fn:Path):
    return [json.loads(l) for l in tqdm(fn.open('r'))]

def save_jsonl(data:list, fn:Path):
    out_str = '\n'.join([json.dumps(d) for d in data])
    fn.open('w').write(out_str)

def get_all_relations(data:dict, splits:list)->list[str]:
    relations = [t['predicate']['surfaceform'] for split in splits for d in data[split] for t in d['triples']]
    return relations

def get_relation_counts(data:dict, splits:list):
    relations = get_all_relations(data, splits)
    counter = Counter(relations)
    return counter

def get_label_to_index(data:dict, split:str='train', num_labels:int=220, add_no_relation:bool=True, no_relation_label:str='has no relation to')->dict[str, int]:
    relations_counter = get_relation_counts(data, [split])
    label_to_index = {label: i for i, (label, _) in enumerate(relations_counter.most_common(num_labels))}
    assert no_relation_label not in label_to_index.keys()
    if add_no_relation:
        label_to_index[no_relation_label] = len(label_to_index)
    return label_to_index

def data_to_surfaceform(data:dict):
    data = {
        split: [
            {
                'triplets': [[t['subject']['surfaceform'], t['predicate']['surfaceform'], t['object']['surfaceform']] for t in d['triples']], 
                'entities': [e['surfaceform'] for e in d['entities']], 
                'title': d['title'],
                'text': d['text']
            } 
            for d in tqdm(data[split])
        ]
        for split in data.keys()
    }
    return data

def data_to_uri(data:dict):
    data = {
        split: [
            {
                'triplets': [[t['subject']['uri'], t['predicate']['uri'], t['object']['uri']] for t in d['triples']], 
                'entities': [e['uri'] for e in d['entities']], 
            } 
            for d in tqdm(data[split])
        ]
        for split in data.keys()
    }
    return data

def load_wikidata_triplets(fn)->pd.DataFrame:
    df = pd.read_csv(fn, sep='\t', header=None, names=['subject', 'predicate', 'object'])
    # df['entities'] = df[['subject', 'object']].apply(lambda x: tuple(sorted(x)), axis=1)
    return df

def get_sparql():
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat('json')
    return sparql

def get_query(ids:list[str]):
    ids_str = 'wd:' + ' wd:'.join(ids)
    query = """
    SELECT ?s ?p ?o
    WHERE {
    VALUES ?s { <IDS> }
    VALUES ?o { <IDS> }
    { ?s ?p ?o }
    }
    """

    query = query.replace('<IDS>', ids_str)
    return query

def get_connecting_triplets(ids:list[str], sparql:SPARQLWrapper):
    """
    returns list of triplets that connect two entities in ids. Each triplet is a list with three elements: subject, predicate, object
    """
    query = get_query(ids)

    sparql.setQuery(query)
    results = sparql.query().convert()
    results = results["results"]["bindings"]
    results = [
        [
            res['s']['value'].replace('http://www.wikidata.org/entity/', ''),
            res['p']['value'].replace('http://www.wikidata.org/prop/direct/', ''),
            res['o']['value'].replace('http://www.wikidata.org/entity/', ''),
        ]
        for res in results
    ]
    return results

def remove_external_links(ids:list[str]):
    ids = [e for e in ids if e.startswith(('P', 'Q', 'L')) and e[1:].isnumeric()]
    return ids

def augment_with_wikidata_triplets(data:dict, sparql:SPARQLWrapper, out_fn:Path):
    """
    Augment with all triplets in wikidata that directly connect two entities in rebel. Works in-place. 
    """
    if out_fn is None:
        num_done_already = 0
    else:
        num_done_already = sum(1 for i in open(out_fn, 'rb')) if out_fn.exists() else 0
    logging.info(f'num_done_already: {num_done_already}')

    for d in tqdm(data[num_done_already:], total=len(data), initial=num_done_already):
        entities = {
            e['uri']: {
                'uri': e['uri'], 
                'boundaries': e['boundaries'],
                'surfaceform': e['surfaceform'],
                'annotator': e['annotator'],
            } 
            for e in d['entities']
        }

        # get all triplets that connect e1 and e2
        ids = remove_external_links(list(entities.keys()))
        tmp_triplets = get_connecting_triplets(ids, sparql)

        d['triples'] = [
            {
                'subject': entities[s],
                'predicate': {'uri': p, 'annotator': 'MP'},
                'object': entities[o]
            }
            for s, p, o in tmp_triplets
        ]
        if out_fn is not None:
            out_fn.open('a').write(json.dumps(d) + '\n')
    return None  # edits in-place and saves directly to file

def get_query_property_surfaceform(idx:str):
    ids_str = f'wd:{idx}'
    query = """
    SELECT ?p ?pLabel
    WHERE {
        VALUES ?p { <IDS> }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    query = query.replace('<IDS>', ids_str)
    return query

def get_surfaceform_of_property(idx:str, sparql:SPARQLWrapper, property_surfaceform:dict, fn_property_surfaceform:Path):
    if idx in property_surfaceform.keys():
        return property_surfaceform[idx]
    logging.debug(f'getting surfaceform of property {idx} from wikidata API')
    query = get_query_property_surfaceform(idx)
    sparql.setQuery(query)
    results = sparql.query().convert()
    result = results["results"]["bindings"][0]["pLabel"]["value"]
    property_surfaceform[idx] = result  # modify in-place
    if fn_property_surfaceform is not None:
        fn_property_surfaceform.open('a').write(idx + '\t' + result + '\n')
    return results

def add_surfaceform_to_predicate(data:dict, sparql:SPARQLWrapper, fn_property_surfaceform:Path):
    property_surfaceform = {l.split('\t')[0]: l.split('\t')[1].strip() for l in fn_property_surfaceform.open('r') if l.strip()}
    for d in tqdm(data):
        del_triplets = []
        for i, t in enumerate(d['triples']):
            idx = t['predicate']['uri']
            if not (idx.startswith('P') and idx[1:].isnumeric()):
                del_triplets.append(i)
                continue
            completed = False
            while not completed:
                try:
                    t['predicate']['surfaceform'] = get_surfaceform_of_property(idx, sparql, property_surfaceform, fn_property_surfaceform)
                    completed = True
                except Exception as e:
                    logging.error(f'idx: {idx}')
                    logging.error(e)
                    logging.error('retrying in 60 sec...')
                    sleep(60)
        for i in reversed(del_triplets):
            del d['triples'][i]
    return None  # edits in-place

def fix_annotator(data_augmented:list, data:list):
    """
    if a triplet occurs in data, then the annotator for that triplet in data_augmented is set to `rebel`. 
    """
    assert len(data_augmented) == len(data), f'{len(data_augmented)} != {len(data)}'
    for d_aug, d in tqdm(zip(data_augmented, data), total=len(data_augmented)):
        data_triples = [(t['subject']['uri'], t['predicate']['uri'], t['object']['uri']) for t in d['triples']]
        for t in d_aug['triples']:
            if (t['subject']['uri'], t['predicate']['uri'], t['object']['uri']) in data_triples:
                t['predicate']['annotator'] = 'rebel'
    return None  # edits in-place

def get_instance(d:dict, label_to_index:dict, add_no_relation:bool, no_relation_label:str, no_relation_prob:float):
    # print(d)
    try:
        instance = {
            'docid': d['docid'],
            'uri': d['uri'],
            'title': d['title'],
            'text': d['text'],
            'triplets': [[t['subject']['surfaceform'], t['predicate']['surfaceform'], t['object']['surfaceform']] for t in d['triples']],
            'predicate_annotations': [t['predicate']['annotator'] for t in d['triples']],
        }
    except Exception as e:
        logging.error(e)
        logging.error(d)
        for i, t in enumerate(d['triples']):
            for what in ['subject', 'predicate', 'object']:
                if not 'surfaceform' in t[what].keys():
                    print(i, what, t[what])
        raise e
    # get triplets that are possible mask candidates
    # we only consider triplets that dont share head and tail with other triplets, and that have a label in label_to_index
    all_existing_ht = [(h,t) for h, _, t in instance['triplets']]
    block_ht = Counter(all_existing_ht)
    block_ht = [ht for ht, c in block_ht.items() if c > 1]
    mask_candidates = [i for i, t in enumerate(instance['triplets']) if t[1] in label_to_index.keys() and not (t[0], t[2]) in block_ht]
    if len(mask_candidates) == 0:
        return None

    if add_no_relation and random.random() < no_relation_prob:
        # get pair of entities that are not connected by any triplet
        all_existing_ht = {frozenset(ht) for ht in all_existing_ht}
        # all_entities = {e['surfaceform'] for e in d['entities']}
        all_entities = {e for ht in all_existing_ht for e in ht}
        all_possible_ht = {frozenset((h,t)) for h in all_entities for t in all_entities if h != t}
        no_relation_candidates = all_possible_ht - all_existing_ht
        if len(no_relation_candidates) == 0:
            return None
        
        no_relation_candidates = [list(ht) for ht in no_relation_candidates]
        no_relation_candidate = random.choice(no_relation_candidates)
        random.shuffle(no_relation_candidate)  # should be random anyways as it comes from a frozenset, but just to be sure

        instance['triplets'].append([no_relation_candidate[0], '<mask>', no_relation_candidate[1]])
        instance['label'] = label_to_index[no_relation_label]
        instance['label_str'] = no_relation_label
        instance['mask_origin'] = 'no_relation'
        instance['predicate_annotations'].append('no_relation')
        return instance
    mask_idx = random.choice(mask_candidates)
    instance['label'] = label_to_index[instance['triplets'][mask_idx][1]]
    instance['label_str'] = instance['triplets'][mask_idx][1]
    instance['triplets'][mask_idx][1] = '<mask>'
    instance['mask_origin'] = 'graph' if instance['predicate_annotations'][mask_idx] == 'MP' else 'text'  # text if annotator is rebel, i.e. the triplet is entailed by the text. graph if annotator is MP (which stands for Moritz Plenz, due to my severe lack of creativity), i.e. the triplet is not entailed by the text and hence has to be inferred from the graph.
    return instance

def data_to_hf_dataset(data:dict, label_to_index:dict, out_fn:Path, add_no_relation:bool, no_relation_label:str, no_relation_prob:float):
    """
    converts data to HF readable dataset. 
    Can be loaded with

    from datasets import load_dataset
    data = load_dataset('json', data_files={split: f'data/rebel_dataset/en_{split}_hf_dataset.jsonl' for split in splits})
    """
    data = [
        get_instance(d, label_to_index, add_no_relation, no_relation_label, no_relation_prob)
        for d in tqdm(data)
    ]
    old_len = len(data)
    data = [d for d in data if d is not None]
    logging.debug(f'removed {old_len - len(data)} instances')
    save_jsonl(data, out_fn)
    return None

def hf_dataset_to_h5py(in_fns:dict[str,str], out_fn, label_to_index:dict[str,int]):
    in_fns = {split: str(fn) for split, fn in in_fns.items()}
    data = load_dataset('json', data_files=in_fns)
    with h5py.File(out_fn, 'w') as f:
        for split in data.keys():
            dst = f.create_dataset(split, (len(data[split]),), dtype=h5py.string_dtype())
            for i, d in tqdm(enumerate(data[split])):
                dst[i] = json.dumps(d)
        f.attrs['num_labels'] = len(label_to_index)
        f.attrs['label_to_index'] = json.dumps(label_to_index)
        f.attrs['num_sources'] = 3
        f.attrs['source_to_index'] = json.dumps({'text': 0, 'graph': 1, 'no_relation': 2})
    return None

def add_onetriplet_dataset_to_h5py(in_fn:str, out_fn:str):
    """
    In the new file, triplets are removed except for triplet that has the masked relation. Thus, not graph information is available in this setting. 
    """
    with h5py.File(out_fn, 'w') as f_out:
        with h5py.File(in_fn, 'r') as f_in:
            for split in f_in.keys():
                logging.info(f'processing {split}')
                dst = f_out.create_dataset(split, (len(f_in[split]),), dtype=h5py.string_dtype())
                for i, d in tqdm(enumerate(f_in[split]), total=len(f_in[split])):
                    d = json.loads(d)
                    d['triplets'] = [t for t in d['triplets'] if t[1] == '<mask>']
                    dst[i] = json.dumps(d)
            f_out.attrs['num_labels'] = f_in.attrs['num_labels']
            f_out.attrs['label_to_index'] = f_in.attrs['label_to_index']
            f_out.attrs['num_sources'] = f_in.attrs['num_sources']
            f_out.attrs['source_to_index'] = f_in.attrs['source_to_index']

def text_entailed_only_to_h5py(in_fn:str, out_fn:str):
    """
    In the new file, only instances where (i) the masked relation is entailed by the text or (ii) the relation is no-relation are included. 
    """
    with h5py.File(out_fn, 'w') as f_out:
        with h5py.File(in_fn, 'r') as f_in:
            for split in f_in.keys():
                logging.info(f'processing {split}')
                num_new_instances = sum(1 for d in f_in[split] if json.loads(d)['mask_origin'] in ['text', 'no_relation'])
                dst = f_out.create_dataset(split, (num_new_instances,), dtype=h5py.string_dtype())
                i = 0
                for d in tqdm(f_in[split], total=len(f_in[split])):
                    d = json.loads(d)
                    if d['mask_origin'] in ['text', 'no_relation']:
                        dst[i] = json.dumps(d)
                        i += 1
            f_out.attrs['num_labels'] = f_in.attrs['num_labels']
            f_out.attrs['label_to_index'] = f_in.attrs['label_to_index']
            f_out.attrs['num_sources'] = f_in.attrs['num_sources']
            f_out.attrs['source_to_index'] = f_in.attrs['source_to_index']

def main(args):
    data = None
    data_augmented = None
    data_augmented_surfaceform = None

    if not (args.label_to_index_fn.exists() and args.index_to_label_fn.exists()):
        if data is None:
            logging.info('load data')
            data = {split: load_jsonl(fn) for split, fn in args.in_fns.items()}
        logging.info('create label_to_index and index_to_label')
        label_to_index = get_label_to_index(data, split='train', num_labels=220, add_no_relation=args.add_no_relation, no_relation_label=args.no_relation_label)
        index_to_label = {i: label for label, i in label_to_index.items()}
        assert len(label_to_index) == len(index_to_label)
        json.dump(label_to_index, args.label_to_index_fn.open('w'), indent=2)
        json.dump(index_to_label, args.index_to_label_fn.open('w'), indent=2)

    # if not args.all_relations_fn.exists():
    #     if data is None:
    #         logging.info('load data')
    #         data = {split: load_jsonl(fn) for split, fn in args.in_fns.items()}
    #     logging.info('create all_relations')
    #     all_relations = get_relation_counts(data=data, splits=['train'])
    #     json.dump(all_relations, args.all_relations_fn.open('w'), indent=2)

    if False:  # augment with wikidata triplets
        if data is None:
            logging.info('load data')
            data = {split: load_jsonl(fn) for split, fn in args.in_fns.items()}
        logging.info('augment with wikidata triplets')
        logging.debug('open sparql connection')
        sparql = get_sparql()
        logging.debug('augment with wikidata triplets')
        for split, data_split in data.items():
            logging.debug(f'augmenting {split}')
            completed = False
            while not completed:
                try:
                    augment_with_wikidata_triplets(data_split, sparql, args.fns_augmented[split])
                    completed = True
                except Exception as e:
                    logging.error(e)
                    logging.error('retrying in 60 sec...')
                    sleep(60)
        data = None
        data_augmented = None

    if False:  # add surfaceform to predicates
        if data is None:
            logging.info('load data')
            data = {split: load_jsonl(fn) for split, fn in args.in_fns.items()}
        if data_augmented is None:
            logging.info('load augmented data')
            data_augmented = {split: load_jsonl(fn) for split, fn in args.fns_augmented.items()}
        logging.info('add surfaceform to predicate')
        sparql = get_sparql()
        for split, data_aug_split in data_augmented.items():
            logging.debug(f'adding surfaceform to predicate in {split}')
            add_surfaceform_to_predicate(data=data_aug_split, sparql=sparql, fn_property_surfaceform=args.fn_property_surfaceform)
            fix_annotator(data_aug_split, data[split])
            save_jsonl(data_aug_split, fn=args.fns_property_surfaceform[split])
        data = None
        data_augmented = None

    if False:  # convert to HF dataset format
        if data_augmented_surfaceform is None:
            logging.info('load augmented data')
            data_augmented_surfaceform = {split: load_jsonl(fn) for split, fn in args.fns_property_surfaceform.items()}
        label_to_index = json.load(args.label_to_index_fn.open('r'))
        logging.info('convert to HF dataset-disk format')
        for split, data_aug_split in data_augmented_surfaceform.items():
            logging.debug(f'converting and saving {split}')
            data_to_hf_dataset(data_aug_split, label_to_index, args.fns_hf_dataset[split], add_no_relation=args.add_no_relation, no_relation_label=args.no_relation_label, no_relation_prob=args.no_relation_prob)
    
    if False:  # convert HF dataset to hdf5
        logging.info('convert HF dataset to hdf5')
        label_to_index = json.load(args.label_to_index_fn.open('r'))
        hf_dataset_to_h5py(args.fns_hf_dataset, args.fn_hdf5, label_to_index)

    if False:  # delete graphs except of triplet which has masked relation
        logging.info('get no_graph data')
        add_onetriplet_dataset_to_h5py(args.fn_hdf5, args.fn_hdf5_tripletonly)

    if False:  # create dataset with only instances where relation is no-relation or where it is entailed by the text. 
        logging.info('create TRIPLET ONLY dataset with only instances where relation is no-relation or where it is entailed by the text')
        text_entailed_only_to_h5py(args.fn_hdf5_tripletonly, args.fn_hdf5_tripletonly_textentailedonly)
    
    if True: # create dataset with only instances where relation is no-relation or where it is entailed by the text. 
        logging.info('create dataset with only instances where relation is no-relation or where it is entailed by the text')
        text_entailed_only_to_h5py(args.fn_hdf5, args.fn_hdf5_textentailedonly)



if __name__ == "__main__":
    args = Args()

    logging.basicConfig(
        level=args.logging_level,
        format=f"%(asctime)s [%(levelname)s] %(filename)s, Line %(lineno)d\n%(message)s",
        datefmt=f"%H:%M:%S",
    )

    random.seed(0)
    main(args)

    print("done with main")
