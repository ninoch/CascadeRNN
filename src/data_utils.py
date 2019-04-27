import os
import codecs
import networkx as nx
import numpy as np
import pickle


def get_node_index(data_dir):
    pkl_path = os.path.join(data_dir, 'node_index.pkl')
    if os.path.isfile(pkl_path):
        print ('>>> Node index pickle exists.')
        node_index = pickle.load(open(pkl_path, 'rb'))
    else:
        # loads cascades
        all_nodes = set()
        filename = os.path.join(data_dir, 'train.txt')
        with codecs.open(filename, 'r', encoding='utf-8') as input_file:
            for line_index, line in enumerate(input_file):
                # parses the input line.
                query, cascade = line.strip().split(' ', 1)
                sequence = [query] + cascade.split(' ')[::2]
                all_nodes.update(sequence)

        filename = os.path.join(data_dir, 'test.txt')
        with codecs.open(filename, 'r', encoding='utf-8') as input_file:
            for line_index, line in enumerate(input_file):
                # parses the input line.
                query, cascade = line.strip().split(' ', 1)
                sequence = [query] + cascade.split(' ')[::2]
                all_nodes.update(sequence)

        all_nodes = list(all_nodes)
        node_index = dict()
        for i, v in enumerate(all_nodes):
            node_index[v] = i
        pickle.dump(node_index, open(pkl_path, 'wb'))

    return node_index


def load_graph(data_dir):
    # loads graph
    graph_file = os.path.join(data_dir, 'graph.txt')
    pkl_file = os.path.join(data_dir, 'graph.pkl')

    node_index = get_node_index(data_dir)

    if os.path.isfile(pkl_file):
        print ('>>> Graph pickle exists.')
        G = pickle.load(open(pkl_file, 'rb'))
    else:
        G = nx.Graph() # It is not DiGraph
        G.name = data_dir
        node_cnt = len(node_index)
        G.add_nodes_from(range(node_cnt))
        with open(graph_file, 'rb') as f:
            for ind, line in enumerate(f):
                if ind == 0:
                    continue
                u, v = line.strip().split()
                if u not in node_index:
                    node_index[u] = node_cnt 
                    node_cnt += 1
                if v not in node_index:
                    node_index[v] = node_cnt
                    node_cnt += 1
                u = node_index[u]
                v = node_index[v]
                G.add_edge(u, v)

        pickle.dump(G, open(pkl_file, 'wb'))

    return G, node_index


def load_examples(data_dir,
                  maxlen,
                  dataset=None,
                  G=None,
                  node_index=None,
                  keep_ratio=1.):
    pkl_path = os.path.join(data_dir, dataset + '.pkl')
    if os.path.isfile(pkl_path):
        print ('>>> Example pickle exists.')
        examples, examples_mask = pickle.load(open(pkl_path, 'rb'))
    else:
        # loads cascades
        filename = os.path.join(data_dir, dataset + '.txt')
        examples = []
        examples_mask = []
        end_of_sequence = max(node_index.values())
        with codecs.open(filename, 'r', encoding='utf-8') as input_file:
            for line_index, line in enumerate(input_file):
                # parses the input line.
                query, cascade = line.strip().split(' ', 1)
                sequence = [query] + cascade.split(' ')[::2]
                mask = []
                if len(sequence) > maxlen:
                    sequence = sequence[:maxlen]
                sequence = [node_index[x] for x in sequence]
                if len(sequence) < maxlen:
                    mask = [1] * (len(sequence) - 1) + [0] * (maxlen - len(sequence))
                    sequence.extend([0] * (maxlen - len(sequence)))
                else:
                    mask = [1] * (maxlen - 1)

                examples.append(sequence)
                examples_mask.append(mask)

        pickle.dump([examples, examples_mask], open(pkl_path, 'wb'))

    n_samples = len(examples)
    indices = np.random.choice(n_samples, int(
        n_samples * keep_ratio), replace=False)
    sampled_examples = [examples[i] for i in indices]
    sampled_masks = [examples_mask[i] for i in indices]
    return np.array(sampled_examples), np.array(sampled_masks)

def load_data(data_dir, maxlen, keep_ratio=1.0):
    G, node_index = load_graph(data_dir)
    train_examples, train_mask = load_examples(data_dir, 
                                  maxlen,
                                  dataset='train',
                                  node_index=node_index,
                                  keep_ratio=keep_ratio,
                                  G=G)
    test_examples, test_mask = load_examples(data_dir, 
                                  maxlen,
                                  dataset='test',
                                  node_index=node_index,
                                  keep_ratio=keep_ratio,
                                  G=G)
    return G, node_index, train_examples, train_mask, test_examples, test_mask 

class Loader:
    def __init__(self, data, data_mask, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.data_mask = data_mask
        self.indices = np.arange(len(self.data), dtype="int32")
        self.idx = 0 

    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def __call__(self):
        if self.idx == 0:
            np.random.shuffle(self.indices)

        batch_indices = self.indices[self.idx: self.idx + self.batch_size]
        batch = np.take(self.data, batch_indices, axis=0)
        batch_mask = np.take(self.data_mask, batch_indices, axis=0)

        self.idx += self.batch_size
        if self.idx >= len(self.data):
            self.idx = 0

        return batch, batch_mask
