import torch
from functools import cache
from argparse import Namespace
from typing import List, Tuple, Dict, Union, Optional
from itertools import chain
import random

from models.graph_T5.graph_t5 import T5Tokenizer
from models.graph_T5.graph_t5.modeling_t5 import T5Attention
import models.graph_T5.graph_t5.modeling_t5

class Graph():
    """
    A graph class.
    :param g: A list of tuples, where each tuple is a triple (head, r, tail).
    """
    def __init__(
            self, 
            g: List[Tuple[str,str,str]] = []
        ):
        self.g = g
        self.concepts = self.get_concepts()  # list of all concepts in the graph
        self.relations = self.get_relations()  # list of all relations in the graph
        self.relations_multiple = self.get_relations_multiple()  # list of all relations in the graph, including duplicate relations

    @property
    def g(self) -> List[Tuple[str,str,str]]:
        return self._g

    @g.setter
    def g(self, g: List[Tuple[str,str,str]]):
        self._g = g

    def num_triplets(self) -> int:
        """
        Get the number of triplets in the graph.
        """
        return len(self.g)

    def get_concepts(self) -> List[str]:
        """
        Get the concepts in the graph.
        """
        concepts = list(set([triplet[i] for triplet in self.g for i in [0, 2]]))
        concepts.sort()  # not necessary but makes debugging easier
        return concepts    
    
    def get_relations(self) -> List[str]:
        """
        Get the relations in the graph.
        """
        relations = list(set(self.get_relations_multiple()))
        relations.sort()  # not necessary but makes debugging easier
        return relations
    
    def get_relations_multiple(self) -> List[str]:
        """
        Get the relations in the graph, including duplicate relations.
        """
        relations = [triplet[1] for triplet in self.g]
        return relations

    def get_neighbors(self, concept:str, radius:int=1) -> List[str]:
        """
        Get the neighbors of a concept.
        :param concept: The concept for which to get the neighbors.
        :param radius: The radius of the neighborhood.
        """
        assert radius >= 1, f"radius={radius} must be >= 1"
        neighbors = []
        for triplet in self.g:
            if concept in triplet:
                neighbors.extend([triplet[i] for i in [0, 2] if triplet[i] != concept])
        if radius > 1:
            neighbors = list(set(neighbors))
            for neighbor in neighbors:
                neighbors.extend(self.get_neighbors(neighbor, radius=radius-1))
        neighbors = list(set(neighbors))
        return neighbors
    
    def mask_neighbors(self, size:int=1) -> None:
        """
        Mask the neighbors around the masked relation.
        :param size: The size of the neighborhood to mask. 1 means only the direct neighbors are masked. 2 means the direct neighbors and following relations are masked. 3 means the direct neighbors, following relations, and following neighbors are masked. Etc.
        """
        assert size >= 1, f"size={size} must be >= 1"

        concepts_to_mask = []
        relation_mask_counter = 0

        for _ in range(size):
            concepts_to_mask_tmp = set()
            for triplet in self.g:
                if triplet[0] == '<mask>' and not triplet[1].startswith('<r_mask_'):
                    triplet[1] = f'<r_mask_{relation_mask_counter}>'
                    relation_mask_counter += 1
                    concepts_to_mask_tmp.add(triplet[0])
                    continue
                if triplet[2] == '<mask>' and not triplet[1].startswith('<r_mask_'):
                    triplet[1] = f'<r_mask_{relation_mask_counter}>'
                    relation_mask_counter += 1
                    concepts_to_mask_tmp.add(triplet[2])
                    continue

                if triplet[1] == '<mask>': 
                    if triplet[0] not in concepts_to_mask:
                        concepts_to_mask_tmp.add(triplet[0])
                    if triplet[2] not in concepts_to_mask:
                        concepts_to_mask_tmp.add(triplet[2])
                    continue

                if triplet[0] in concepts_to_mask and triplet[1].startswith('<r_mask_'):
                    if triplet[2] not in concepts_to_mask:
                        concepts_to_mask_tmp.add(triplet[2])
                    continue

                if triplet[2] in concepts_to_mask and triplet[1].startswith('<r_mask_'):
                    if triplet[0] not in concepts_to_mask:
                        concepts_to_mask_tmp.add(triplet[0])
                    continue

                if triplet[0] in concepts_to_mask or triplet[2] in concepts_to_mask:
                    triplet[1] = f'<r_mask_{relation_mask_counter}>'
                    relation_mask_counter += 1
                    continue
            
            concepts_to_mask_tmp = concepts_to_mask_tmp - set(concepts_to_mask)
            concepts_to_mask.extend(concepts_to_mask_tmp)
            print(_, concepts_to_mask, concepts_to_mask_tmp)

        if '<mask>' in concepts_to_mask:
            concepts_to_mask.remove('<mask>')  # should remain normal "<mask>"
        assert len(concepts_to_mask) == len(set(concepts_to_mask))
        concept_to_num = {concept: i for i, concept in enumerate(concepts_to_mask)}
        for triplet in self.g:
            if triplet[0] in concepts_to_mask:
                triplet[0] = f'<c_mask_{concept_to_num[triplet[0]]}>'
            if triplet[2] in concepts_to_mask:
                triplet[2] = f'<c_mask_{concept_to_num[triplet[2]]}>'

    def __str__(self):
        out_str = '\n'.join([str(triplet) for triplet in self.g])        
        return out_str
    
    def __eq__(self, other): 
        if self.__class__ != other.__class__: 
            return False
        return self.g == other.g

class Data(Namespace):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

def get_dummy_graph(num_triplets:int=3) -> Graph:
    g = [
        ("dog", "IsA", "animal"),
        ("cat", "IsA", "animal"),
        ("black poodle", "IsA", "dog"),
        ("black cat", "IsA", "cat"),
    ]
    assert num_triplets <=4, "num_triplets must be <= 4"
    g = g[:num_triplets]
    g = Graph(g)
    return g

@cache
def get_r2nl_dict() -> dict:
    relations = {
    # ConceptNet
    "/r/RelatedTo": "is related to",
    "/r/IsA": "is a",
    "/r/FormOf": "is a form of",
    "/r/CapableOf": "is capable of",
    "/r/MotivatedByGoal": "is motivated by",
    "/r/HasContext": "has context",
    "/r/HasPrerequisite": "has prerequisite",
    "/r/Synonym": "is a synonym of",
    "/r/Antonym": "is an antonym of",
    "/r/AtLocation": "is in",
    "/r/Desires": "desires",
    "/r/UsedFor": "is used for",
    "/r/HasSubevent": "has subevent",
    "/r/HasProperty": "is",
    "/r/PartOf": "is a part of",
    "/r/DefinedAs": "is defined as",
    "/r/HasA": "has",
    "/r/MannerOf": "is a manner of",
    "/r/Causes": "causes",
    "/r/HasFirstSubevent": "starts with",
    "/r/HasLastSubevent": "ends with",
    "/r/ReceivesAction": "receives action",
    "/r/InstanceOf": "is an instance of",
    "/r/NotCapableOf": "is not capable of",
    "/r/CausesDesire": "causes desire",
    "/r/DistinctFrom": "is distinct from",
    "/r/NotDesires": "does not desire",
    "/r/MadeOf": "is made of",
    "/r/Entails": "entails",
    "/r/CreatedBy": "is created by",
    "/r/NotHasProperty": "is not",
    "/r/LocatedNear": "is near",
    "/r/SymbolOf": "is a symbol of",

    # mask token of T5
    "<mask>": "<extra_id_0>",
    }
    relation_masks = {f'<r_mask_{i}>': f'<extra_id_{i+1}>' for i in range(0,49)}
    concept_masks = {f'<c_mask_{i}>': f'<extra_id_{i+50}>' for i in range(0,50)}
    
    r2nl_dict = dict()
    r2nl_dict.update(relations)
    r2nl_dict.update(relation_masks)
    r2nl_dict.update(concept_masks)
    return r2nl_dict

def r2nl(r: str) -> str:
    """
    Convert a relation to a natural language string.
    """
    r2nl_dict = get_r2nl_dict()  # get_r2nl_dict is cached

    if r in r2nl_dict.keys():
        return r2nl_dict[r]
    else:
        return r

def _get_str2tok(g:Graph, tokenizer: T5Tokenizer) -> dict[str, list[int]]:
    """
    Get a dictionary that maps strings to tokens.
    """
    # tokenize concepts and relations
    c_tok = tokenizer([r2nl(c) for c in g.concepts], padding=False)['input_ids']
    r_tok = tokenizer([r2nl(r) for r in g.relations], padding=False)['input_ids']

    tokens = c_tok + r_tok
    node_names = g.concepts + g.relations  # these are not necessarily all nodes in the Levi Graph, as relations can occur more than once
    assert len(tokens) == len(node_names), f"{len(tokens) = }, {len(node_names) = }"

    # remove end-of-sequence token
    tokens = [toks[:-1] if toks[-1] == tokenizer.eos_token_id else toks for toks in tokens]
    
    # create a dictionary mapping concepts and relations to their tokenized forms
    str2tok = {node: tok for node, tok in zip(node_names, tokens)}
    str2tok['</s>'] = [tokenizer.eos_token_id]
    return str2tok

def _get_graphT5_input_sequence(g:Graph, str2tok:dict, use_eos:bool) -> Tuple[list, dict]:
    # get input sequence (i.e. sequence that will be fed into the model for this graph)
    all_nodes = g.relations_multiple + g.concepts  # list of all concepts and relations that will be in the final sequence (i.e. all nodes of the Levi Graph)  # the order of nodes is first all relations (in the order that they appear in g.g), and then all concepts (in alphabetical order. though here the order is not important)
    
    if use_eos:
        all_nodes.append('</s>')

    all_tokens = [str2tok[node] for node in all_nodes]  # list of length #nodes, where each element is a list of token ids
    indices = {node: [] for node in all_nodes}  # dictionary mapping each node to its start-index and end- in the sequence. Keys are nodes, values are lists of tuples (start_index, end_index). The lists have a length of 1 for concepts and are as long as the number of occurances of the relation in the graph for relations.  # WARNING: this assumes that concepts and realtions have different names. This not always the case for REBEL. For concept_indices this is fixed. 
    num_relation_tokens = sum([len(token) for token in all_tokens[:len(g.relations_multiple)]])  # number of tokens that are relations
    num_concept_tokens = sum([len(token) for token in all_tokens[len(g.relations_multiple):len(g.relations_multiple)+len(g.concepts)]])  # number of tokens that are concepts
    num_eos_tokens = 1 if use_eos else 0

    is_concept = torch.tensor([False] * num_relation_tokens + [True] * num_concept_tokens + [False] * num_eos_tokens, dtype=torch.bool)  # tensor of length #nodes, where each element is True if the node is a concept and False if it is a relation
    index_counter = 0
    assert len(all_nodes) == len(all_tokens), (all_nodes, all_tokens)

    for node, token in zip(all_nodes, all_tokens):
        indices[node].append((index_counter, index_counter + len(token)))
        # assert is_concept[index_counter:index_counter+len(token)].all() == (node in g.concepts), f"{is_concept = }, {node = }, {g.concepts = }, {index_counter = }, {len(token) = }, {is_concept[index_counter:index_counter+len(token)] = }"
        index_counter += len(token)

    concept_indices = {node: [indices[node][-1]] for node in g.concepts}  # [-1] and reput in list in case relations have the same name as a concept (concepts are put in last). 
    sequence = torch.tensor(list(chain.from_iterable(all_tokens)), dtype=torch.long)
    sequence = sequence.unsqueeze(0)  # add batch dimension
    is_concept = is_concept.unsqueeze(0)  # add batch dimension
    return sequence, indices, is_concept, concept_indices

def _get_graphT5_relativeposition_sparsitymask(g:Graph, indices:dict, sequence_length:int, use_eos:bool, eos:str) -> Tuple[torch.Tensor, torch.Tensor]:
    ### get relative position of each node in the sequence, as well as the sparsity mask ###
    # initialize relative position matrix)
    relative_position = torch.zeros(size=(sequence_length, sequence_length), dtype=torch.long) 
    # initialize sparsity mask
    sparsity_mask = torch.zeros(size=(sequence_length, sequence_length), dtype=torch.bool)
    # initialize use_additional_bucket
    use_additional_bucket = torch.zeros(size=(sequence_length, sequence_length), dtype=torch.bool)
    
    # relative positions / sparsity within each node
    for start, end in chain.from_iterable(indices.values()):
        relative_position[start:end, start:end] = _get_relative_position(end-start)
        sparsity_mask[start:end, start:end] = True

    # relative position between nodes of the same triplet
    relation_counter = {relation: 0 for relation in g.relations}  # dictionary mapping each relation to the number of times it has already appeared in the graph
    for triplet in g.g:
        pos_h = indices[triplet[0]][0]  # position of head; tuple (start_index, end_index)
        pos_r = indices[triplet[1]][relation_counter[triplet[1]]]  # position of relation; tuple (start_index, end_index)
        pos_t = indices[triplet[2]][0]  # position of tail; tuple (start_index, end_index)
        
        l_h, l_r = pos_h[1] - pos_h[0], pos_r[1] - pos_r[0]  # length (i.e. number of tokens) of head and relation

        # iterate over all combinations of tokens in each triplet. This implementation is not very elegant, but it is sufficiently fast.
        for ih, ph in enumerate(range(pos_h[0], pos_h[1])):  # iterate over all head tokens
            for ir, pr in enumerate(range(pos_r[0], pos_r[1])):  # iterate over all relation tokens
                relative_position[ph, pr] = l_h - ih + ir
                relative_position[pr, ph] = - (l_h - ih + ir)
                sparsity_mask[ph, pr] = True
                sparsity_mask[pr, ph] = True
            for it, pt in enumerate(range(pos_t[0], pos_t[1])):  # iterate over all tail tokens
                relative_position[ph, pt] = l_h - ih + l_r + it
                relative_position[pt, ph] = - (l_h - ih + l_r + it)
                sparsity_mask[ph, pt] = True
                sparsity_mask[pt, ph] = True
        for ir, pr in enumerate(range(pos_r[0], pos_r[1])):  # iterate over all relation tokens
            for it, pt in enumerate(range(pos_t[0], pos_t[1])):  # iterate over all tail tokens
                relative_position[pr, pt] = l_r - ir + it
                relative_position[pt, pr] = - (l_r - ir + it)
                sparsity_mask[pr, pt] = True
                sparsity_mask[pt, pr] = True

        relation_counter[triplet[1]] += 1  # next time when that relation comes, then the next tokens will be used
    
    if use_eos:
        assert len(indices['</s>']) == 1, f"{indices['</s>'] = } should have length 1"
        pos_eos = indices['</s>'][0]  # position of head; tuple (start_index, end_index)
        assert pos_eos[0] + 1 == pos_eos[1], pos_eos
        pos_eos = pos_eos[0]  # position of eos token

        if eos == 'bidirectional':
            relative_position[:, pos_eos] = +1e6
            relative_position[pos_eos, :] = -1e6
            relative_position[pos_eos, pos_eos] = 0
            sparsity_mask[:, pos_eos] = True
            sparsity_mask[pos_eos, :] = True
        elif eos == 'unidirectional':
            relative_position[:, pos_eos] = 1e6
            relative_position[pos_eos, pos_eos] = 0
            sparsity_mask[pos_eos, :] = False  # no messages from eos to other tokens
            sparsity_mask[:, pos_eos] = True
        else:
            raise ValueError(f'{eos = } is not a valid option.')
    
    relative_position = relative_position.unsqueeze(0)  # add batch dimension
    sparsity_mask = sparsity_mask.unsqueeze(0)  # add batch dimension
    use_additional_bucket = use_additional_bucket.unsqueeze(0)  # add batch dimension
    return relative_position, sparsity_mask, use_additional_bucket

def _get_global_graphT5_relativeposition_sparsitymask(g:Graph, indices:dict, sequence_length:int, use_eos:bool, eos:str) -> Tuple[torch.Tensor, torch.Tensor]:
    ### get relative position of each node in the sequence, as well as the sparsity mask ###
    # initialize relative position matrix)
    # relative_position = torch.ones(size=(sequence_length, sequence_length), dtype=torch.long) * 1e6  # technically should be float('inf'), but it does not matter
    relative_position = torch.zeros(size=(sequence_length, sequence_length), dtype=torch.long)
    # initialize sparsity mask
    sparsity_mask = torch.ones(size=(sequence_length, sequence_length), dtype=torch.bool)  # could switch to None, but then code has to be updated accordingly (in particular get_batch)
    # initialize use_additional_bucket
    use_additional_bucket = torch.ones(size=(sequence_length, sequence_length), dtype=torch.bool)

    # relative positions / sparsity within each node
    for start, end in chain.from_iterable(indices.values()):
        relative_position[start:end, start:end] = _get_relative_position(end-start)
        use_additional_bucket[start:end, start:end] = False

    # relative position between nodes of the same triplet
    relation_counter = {relation: 0 for relation in g.relations}  # dictionary mapping each relation to the number of times it has already appeared in the graph
    for triplet in g.g:
        pos_h = indices[triplet[0]][0]  # position of head; tuple (start_index, end_index)
        pos_r = indices[triplet[1]][relation_counter[triplet[1]]]  # position of relation; tuple (start_index, end_index)
        pos_t = indices[triplet[2]][0]  # position of tail; tuple (start_index, end_index)
        
        l_h, l_r = pos_h[1] - pos_h[0], pos_r[1] - pos_r[0]  # length (i.e. number of tokens) of head and relation

        # iterate over all combinations of tokens in each triplet. This implementation is not very elegant, but it works.
        for ih, ph in enumerate(range(pos_h[0], pos_h[1])):  # iterate over all head tokens
            for ir, pr in enumerate(range(pos_r[0], pos_r[1])):  # iterate over all relation tokens
                relative_position[ph, pr] = l_h - ih + ir
                relative_position[pr, ph] = - (l_h - ih + ir)
                use_additional_bucket[ph, pr] = False
                use_additional_bucket[pr, ph] = False
            for it, pt in enumerate(range(pos_t[0], pos_t[1])):  # iterate over all tail tokens
                relative_position[ph, pt] = l_h - ih + l_r + it
                relative_position[pt, ph] = - (l_h - ih + l_r + it)
                use_additional_bucket[ph, pt] = False
                use_additional_bucket[pt, ph] = False
        for ir, pr in enumerate(range(pos_r[0], pos_r[1])):  # iterate over all relation tokens
            for it, pt in enumerate(range(pos_t[0], pos_t[1])):  # iterate over all tail tokens
                relative_position[pr, pt] = l_r - ir + it
                relative_position[pt, pr] = - (l_r - ir + it)
                use_additional_bucket[pr, pt] = False
                use_additional_bucket[pt, pr] = False

        relation_counter[triplet[1]] += 1  # next time when that relation comes, then the next tokens will be used
        if use_eos:
            assert len(indices['</s>']) == 1, f"{indices['</s>'] = } should have length 1"
            pos_eos = indices['</s>'][0]  # position of head; tuple (start_index, end_index)
            assert pos_eos[0] + 1 == pos_eos[1], pos_eos
            pos_eos = pos_eos[0]  # position of eos token

            if eos == 'bidirectional':
                relative_position[:, pos_eos] = +1e6
                relative_position[pos_eos, :] = -1e6
                relative_position[pos_eos, pos_eos] = 0
                sparsity_mask[:, pos_eos] = True
                sparsity_mask[pos_eos, :] = True
                use_additional_bucket[:, pos_eos] = False
                use_additional_bucket[pos_eos, :] = False
            elif eos == 'unidirectional':
                relative_position[:, pos_eos] = 1e6
                relative_position[pos_eos, pos_eos] = 0
                sparsity_mask[pos_eos, :] = False  # no messages from eos to other tokens
                sparsity_mask[:, pos_eos] = True
                use_additional_bucket[:, pos_eos] = False
                use_additional_bucket[pos_eos, :] = False
            else:
                raise ValueError(f'{eos = } is not a valid option.')
    
    relative_position = relative_position.unsqueeze(0)  # add batch dimension
    sparsity_mask = sparsity_mask.unsqueeze(0)  # add batch dimension
    use_additional_bucket = use_additional_bucket.unsqueeze(0)  # add batch dimension
    return relative_position, sparsity_mask, use_additional_bucket

def graph_to_graphT5(g: Graph, tokenizer: T5Tokenizer, how:str, eos:str)->Data:
    """
    Convert a graph to a graphT5 input.
    :param g: graph
    :param tokenizer: tokenizer
    :param how: how to represent the graph. Can be 'local' or 'global' for lGLM and gGLM respectively.
    :param eos: end-of-sequence token. Can be `False` for not using an eos token. When using an eos token, there are two ways to use it: `bidirectional` means that the eos token is connected to every other node in the graph, with a relative position of positive infinity (from node to eos) or negative infinity (from eos to node). `unidirectional` means that the eos token is connected to every node in the graph with a relative position of positive infinity (from node to eos), but not the other way around (i.e. no connection from eos to other node). This means, that nodes do not get messages from the eos token, which perceives locality when using the local GLM
    """
    eos = str(eos)
    assert eos in ['False', 'bidirectional', 'unidirectional'], f"{eos = } must be either 'False', 'bidirectional', or 'unidirectional'"
    use_eos:bool = eos != 'False'

    str2tok = _get_str2tok(g, tokenizer)  # get a dictionary mapping concepts and relations to their tokenized forms

    sequence, indices, is_concept, concept_indices = _get_graphT5_input_sequence(g, str2tok, use_eos)  # get input sequence (i.e. sequence that will be fed into the model for this graph
    sequence_length = sequence.shape[1]

    if how == 'local':
        relative_position, sparsity_mask, use_additional_bucket = _get_graphT5_relativeposition_sparsitymask(g, indices, sequence_length, use_eos, eos)
        num_additional_buckets = 0  # lGLM does not use additional buckets
    elif how == 'global':
        relative_position, sparsity_mask, use_additional_bucket = _get_global_graphT5_relativeposition_sparsitymask(g, indices, sequence_length, use_eos, eos)
        num_additional_buckets = 1  # gGLM uses 1 additional bucket for long-ranged G2G connections
    else:
        raise ValueError(f"how must be either 'local' or 'global', but is {how}")

    input_ids = sequence

    data = Data(input_ids=input_ids, relative_position=relative_position, sparsity_mask=sparsity_mask, use_additional_bucket=use_additional_bucket, indices=indices, is_concept=is_concept, concept_indices=concept_indices, num_additional_buckets=num_additional_buckets)

    return data

def _get_set_of_triplets_input_sequence(g:Graph, str2tok:dict, order:str) -> Tuple[list, dict]:
    # get input sequence (i.e. sequence that will be fed into the model for this graph)
    all_nodes = g.relations_multiple + g.concepts  # list of all concepts and relations that will be in the final sequence (i.e. all nodes of the Levi Graph)  # the order of nodes is first all relations (in the order that they appear in g.g), and then all concepts (in alphabetical order. though here the order is not important)
    all_nodes.append('</s>')

    indices = {node: [] for node in all_nodes}  # dictionary mapping each node to its start-index and end- in the sequence. Keys are nodes, values are lists of tuples (start_index, end_index). The lists have a length of #degree(concept) for concepts and are as long as the number of occurances of the relation in the graph for relations.

    index_counter = 0
    sequence = []
    is_concept = [] # list of length #tokens, where each element is True if the token is a concept and False if it is a relation

    if order == 'random':
        generator = random.sample(g.g, len(g.g))
    elif order == 'alphabetical':
        generator = sorted(g.g, key=lambda x: x[1])
    else:
        raise ValueError(f"order must be either 'random' or 'alphabetical', but is {order}")
    
    for triplet in generator:
        for n_i, node in enumerate(triplet):
            token = str2tok[node]
            sequence.extend(token)
            is_concept.extend([n_i != 1] * len(token))
            indices[node].append((index_counter, index_counter + len(token)))
            index_counter += len(token)
        # triplets are separated by </s>
        node = '</s>'
        token = str2tok[node]
        sequence.extend(token)
        is_concept.extend([False] * len(token))
        indices[node].append((index_counter, index_counter + len(token)))
        index_counter += len(token)
    concept_indices = {node: indices[node] for node in g.concepts}
    sequence = torch.tensor(sequence, dtype=torch.long)
    sequence = sequence.unsqueeze(0)  # add batch dimension
    is_concept = torch.tensor(is_concept, dtype=torch.bool)
    is_concept = is_concept.unsqueeze(0)  # add batch dimension
    return sequence, indices, is_concept, concept_indices

def graph_to_set_of_triplets(g:Graph, tokenizer:T5Tokenizer, order:str='random')->Data:
    """
    Convert graph to a T5 input where graph as represented as a set of triplets. Triplets are separated by </s>.
    :param g: graph
    :param tokenizer: tokenizer
    :param order: order of triplets. Can be 'random' or 'alphabetical'
    """
    str2tok = _get_str2tok(g, tokenizer)  # get a dictionary mapping concepts and relations to their tokenized forms
    
    sequence, indices, is_concept, concept_indices = _get_set_of_triplets_input_sequence(g, str2tok, order=order)  # get input sequence (i.e. sequence that will be fed into the model for this graph

    input_ids = sequence
    relative_position = None  # None means that the encoder is a normal sequence transformer
    sparsity_mask = None  # None means that the encoder is a normal sequence transformer
    use_additional_bucket = None  # None means that the encoder is a normal sequence transformer
    num_additional_buckets = 0  # set and list don't use additional buckets

    data = Data(input_ids=input_ids, relative_position=relative_position, sparsity_mask=sparsity_mask, use_additional_bucket=use_additional_bucket, indices=indices, is_concept=is_concept, concept_indices=concept_indices, num_additional_buckets=num_additional_buckets)
    return data

@cache
def _get_relative_position(size):
    return torch.tensor([[i - j for i in range(size)] for j in range(size)], dtype=torch.long)

def get_embedding(
        sequence_embedding: torch.Tensor,
        indices: Dict[str, List[Tuple[int, int]]],
        concept: str,
        embedding_aggregation: str = "mean",
    ):
    """
    Returns the embedding of a concept.
    :param sequence_embedding: the embedding of the whole sequence. shape: (sequence_length, embedding_size)
    :param indices: dictionary mapping each node to its start-index and end- in the sequence. Keys are nodes, values are lists of tuples (start_index, end_index). The lists have a length of 1 for concepts.
    :param concept: the concept for which the embedding should be returned
    :param embedding_aggregation: how the embedding of a concept should be aggregated. Either "mean" or "seq". "mean" returns the mean of all tokens of the concept. "seq" returns the embeddings of the all token of the concept.
    :return: the aggregated embedding of the concept. shape (1, embedding_size) or (number_of_tokens, embedding_size). 
    """
    assert concept in indices.keys(), f"{concept = } is not a node in the graph. {indices = }"
    assert len(indices[concept]) == 1, f"{concept = } is not a concept, as concepts occur only once in the graph. {indices = }"

    start, end = indices[concept][0]
    sequence_embedding = sequence_embedding[start:end, :]
    if embedding_aggregation == "mean":
        return torch.mean(sequence_embedding, dim=0, keepdim=True)
    elif embedding_aggregation == "seq":
        return sequence_embedding
    else:
        raise NotImplementedError(f"{embedding_aggregation = } is not supported. Use either 'mean' or 'seq'.")

def add_text_to_graph_data(data, text, tokenizer, use_text):
    if use_text in {'False', '', False, None}:
        return None

    text_seq = torch.tensor(tokenizer(text, padding=False)['input_ids']).unsqueeze(0)
    new_input_ids = torch.cat([data.input_ids, text_seq], dim=1)

    old_seq_len = data.input_ids.shape[1]
    text_seq_len = text_seq.shape[1]
    new_seq_len = new_input_ids.shape[1]

    new_is_graph = torch.zeros(size=(1, new_seq_len), dtype=torch.bool)
    new_is_graph[:, :old_seq_len] = True

    if data.relative_position is None:  # sequence transformer
        assert data.sparsity_mask is None
        assert data.use_additional_bucket is None
        data.input_ids = new_input_ids
        data.is_graph = new_is_graph
        return None

    new_relative_position = torch.zeros(size=(1, new_seq_len, new_seq_len), dtype=data.relative_position.dtype)
    new_relative_position[:, :old_seq_len, :old_seq_len] = data.relative_position
    new_relative_position[:, old_seq_len:, old_seq_len:] = _get_relative_position(text_seq_len)

    new_sparsity_mask = torch.zeros(size=(1, new_seq_len, new_seq_len), dtype=data.sparsity_mask.dtype)
    new_sparsity_mask[:, :old_seq_len, :old_seq_len] = data.sparsity_mask
    new_sparsity_mask[:, old_seq_len:, old_seq_len:] = True
    
    new_use_additional_bucket = torch.zeros(size=(1, new_seq_len, new_seq_len), dtype=data.use_additional_bucket.dtype)
    new_use_additional_bucket[:, :old_seq_len, :old_seq_len] = data.use_additional_bucket
    new_use_additional_bucket[:, old_seq_len:, old_seq_len:] = False  # could change that if we want T2T and local G2G relations to be learned separately

    data.__dict__.keys()

    if use_text == 'FullyConnected':
        new_sparsity_mask[:, old_seq_len:, :old_seq_len] = True
        new_sparsity_mask[:, :old_seq_len, old_seq_len:] = True

        new_use_additional_bucket[:, old_seq_len:, :old_seq_len] = True
        new_use_additional_bucket[:, :old_seq_len, old_seq_len:] = True

        new_relative_position[:, old_seq_len:, :old_seq_len] = data.num_additional_buckets
        new_relative_position[:, :old_seq_len, old_seq_len:] = data.num_additional_buckets + 1
        
        new_num_additional_buckets = data.num_additional_buckets + 2
    else:
        raise ValueError(f"unknown use_text {use_text} (type {type(use_text)})")

    data.input_ids = new_input_ids
    data.relative_position = new_relative_position
    data.sparsity_mask = new_sparsity_mask
    data.use_additional_bucket = new_use_additional_bucket
    data.num_additional_buckets = new_num_additional_buckets
    data.is_graph = new_is_graph
    return None

