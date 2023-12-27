import os
import torch

from pydgn.data.dataset import TemporalDatasetInterface
from torch_geometric.data import download_url, extract_tar, extract_gz, Data
from torch_geometric.utils import negative_sampling, degree
from os.path import join, isfile, isdir
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# **** LINK PREDICTION ON EDGE/NODE-DYNAMIC GRAPHS ****
class AutonomousSystemsDatasetInterface(TemporalDatasetInterface):
    """
    Autonomous systems AS-733 interface.
    It contains a sequece of graph snapshots for link prediction.
    Available at https://snap.stanford.edu/data/as-733.html
    """

    def __init__(self, root, name='as_733'):
        self.name = name
        self.root = root
        path = self._check_and_download()
        if not isfile(path + '.pt'):
            self.dataset = self._load_data(path)
        else:
            self.dataset = torch.load(path + '.pt')

    def _check_and_download(self):
        path = join(self.root, self.name)
        if not isdir(path):
            tar_file = download_url(f'https://snap.stanford.edu/data/as-733.tar.gz', self.root)
            extract_tar(tar_file, join(self.root, self.name), mode='r:gz')
            os.unlink(tar_file)
        return path

    def _load_data(self, path):
        data_list = []

        path = join(self.root, self.name)
        graph_paths = sorted([join(path, f) for f in os.listdir(path) if isfile(join(path, f))])

        # Map to continuous node id
        nodes_ids = set()
        for i in range(len(graph_paths)):
            g = pd.read_csv(graph_paths[i], skiprows=4, sep='\t', names=['from', 'to'])
            nodes_ids.update(g['from'].values)
            nodes_ids.update(g['to'].values)
        map_id = {old_id: i for i, old_id in enumerate(nodes_ids)}

        g = None
        for i in tqdm(range(len(graph_paths)-1)):
            if g is None:
                g = pd.read_csv(graph_paths[i], skiprows=4, sep='\t', names=['from', 'to'])
                g = g.applymap(lambda x: map_id[x]) # Map to continuous node id
                f = open(graph_paths[i], 'r')
                stats = f.readlines()[2]
                f.close()
                num_nodes = int(stats.split('\t')[0].split(':')[1])

            g_next = pd.read_csv(graph_paths[i+1], skiprows=4, sep='\t', names=['from', 'to'])
            g_next = g_next.applymap(lambda x: map_id[x]) # Map to continuous node id
            num_nodes_ = len(map_id) # By setting the maximum number of nodes, we consider them fixed along time

            edge_index = torch.from_numpy(g.to_numpy().T)
            x = torch.ones(num_nodes_, 1) # This representation has been taken from EvolveGCN

            f = open(graph_paths[i+1], 'r')
            stats_next = f.readlines()[2]
            f.close()
            num_nodes_next = int(stats_next.split('\t')[0].split(':')[1])
            edge_index_next = torch.from_numpy(g_next.to_numpy().T)

            # Negative sampling:
            # Samples random negative edges of a graph given by the graph at time i+1
            neg_edge_index = negative_sampling(edge_index = edge_index_next,
                                               num_nodes = num_nodes_next)

            neg_edge_index = torch.cat((neg_edge_index, torch.zeros(1, neg_edge_index.size(1))))
            edge_index_next = torch.cat((edge_index_next, torch.ones(1, edge_index_next.size(1))))
            target_edge_index = torch.cat((neg_edge_index, edge_index_next), 1)

            ## Remove nodes that are not in the current graph
            #known_node_ids = []
            #for i, (source, target, value) in enumerate(target_edge_index.T):
            #    if source < num_nodes and target < num_nodes:
            #        known_node_ids.append(i)
            #target_edge_index = target_edge_index[:, known_node_ids].type(torch.LongTensor)

            relation_type = torch.zeros(edge_index.shape[1])
            data_list.append(Data(x=x, 
                                  edge_index=edge_index,
                                  y=target_edge_index[2].unsqueeze(-1).type(torch.FloatTensor),
                                  relation_type = relation_type,
                                  link_pred_ids=target_edge_index[:-1].type(torch.LongTensor)))

            g = g_next
            stats = stats_next
            num_nodes = num_nodes_next

        torch.save(data_list, path + '.pt')

        return data_list

    def __getitem__(self, idx):
        data = self.dataset[idx]
        setattr(data, 'mask', self.get_mask(data))
        return data

    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is link prediction at each time step, so the mask is always true
        mask = np.ones((1,1))  #  time_steps x 1
        return mask

    @property
    def dim_node_features(self):
        return self.dataset[0].x.shape[-1]

    @property
    def dim_edge_features(self):
        return 1

    @property
    def dim_target(self):
        # binary link prediction
        return 1

    def __len__(self):
        return len(self.dataset)



class BitcoinAlphaDatasetInterface(TemporalDatasetInterface):
    """
    Bitcoin Alpha network interface for link prediction on discrete-time dynamic graphs.
    It contains a who-trusts-whom network of people who trade using Bitcoin on a platform 
    called Bitcoin Alpha.
    Available at https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html
    """

    def __init__(self, root, name='bitcoin_alpha'):
        self.name = name
        self.root = root
        path = self._check_and_download()
        if not isfile(path + '.pt'):
            self.dataset = self._load_data(path)
        else:
            self.dataset = torch.load(path + '.pt')

    def _check_and_download(self):
        path = join(self.root, self.name)
        if not isdir(path):
            gz_file = download_url(f'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz', self.root)
            extract_gz(gz_file, self.root)
            os.unlink(gz_file)
            extracted_gz = join(self.root, 'soc-sign-bitcoinalpha.csv')
            os.rename(extracted_gz, join(self.root, self.name + '.csv'))
        return path

    def _load_data(self, path):
        data_list = []

        path = join(self.root, self.name)
        df = pd.read_csv(path + '.csv', names=['SOURCE', 'TARGET', 'RATING', 'TIME'])

        # Convert the timestamp to year-month-day
        convert_date = lambda x: str(datetime.fromtimestamp(x)).split(' ')[0] 
        df.TIME = pd.to_datetime(df.TIME.apply(convert_date))

        # Map to continuous node id
        nodes_ids = set()
        nodes_ids.update(df['SOURCE'].values)
        nodes_ids.update(df['TARGET'].values)
        map_id = {old_id: i for i, old_id in enumerate(nodes_ids)}
        df[['SOURCE', 'TARGET']] = df[['SOURCE', 'TARGET']].applymap(lambda x: map_id[x])
        num_nodes = len(map_id)

        # Daily aggregation of snapshots 
        data_list = []
        for _, g in tqdm(df.groupby('TIME')):
            edge_index = torch.from_numpy(g[['SOURCE', 'TARGET']].to_numpy().T)
            x = torch.ones(num_nodes, 1) # This representation has been taken from EvolveGCN
            #x = degree(edge_index[0], num_nodes)
            #x /= x.max()
            #x = x.view(-1, 1)
            data_list.append(Data(x=x, edge_index=edge_index))


        for i in tqdm(range(len(data_list)-1)):
            edge_index = data_list[i].edge_index

            next_edge_index = data_list[i+1].edge_index
            # Negative sampling:
            # Samples random negative edges of a graph given by the graph at time i+1
            neg_edge_index = negative_sampling(edge_index = next_edge_index,
                                                num_nodes = num_nodes)

            neg_edge_index = torch.cat((neg_edge_index, torch.zeros(1, neg_edge_index.size(1))))
            next_edge_index = torch.cat((next_edge_index, 
                                            torch.ones(1, data_list[i+1].edge_index.size(1))))
            target_edge_index = torch.cat((neg_edge_index, next_edge_index), 1).type(torch.LongTensor)

            relation_type = torch.zeros(edge_index.shape[1])
            setattr(data_list[i], 'relation_type', relation_type)
            setattr(data_list[i], 'y', target_edge_index[2].unsqueeze(-1).type(torch.FloatTensor))
            setattr(data_list[i], 'link_pred_ids', target_edge_index[:-1])

        torch.save(data_list[:-1], path + '.pt')

        return data_list

    def __getitem__(self, idx):
        data = self.dataset[idx]
        setattr(data, 'mask', self.get_mask(data))
        return data

    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is link prediction at each time step, so the mask is always true
        mask = np.ones((1,1))  #  time_steps x 1
        return mask

    @property
    def dim_node_features(self):
        return self.dataset[0].x.shape[-1]

    @property
    def dim_edge_features(self):
        return 1

    @property
    def dim_target(self):
        # binary link prediction
        return 1

    def __len__(self):
        return len(self.dataset)
