#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import os.path as osp
import itertools

import os
import json
import tqdm

import torch
from ogb.linkproppred import PygLinkPropPredDataset
from scipy.sparse import csr_matrix
from torch_sparse import coalesce
from tqdm import tqdm
from collections import Counter
import re

from utils import get_pos_neg_edges, np_sampling


class Yelp():
    def __init__(self):
        def parse_yelp(dir):
            """
            Read the yelp dataset from .tar file
            :param dir: the path to raw tar file (yelp_dataset.tar)
            :return: yelp_business, yelp_review, yelp_user, yelp_checkin, yelp_tip, pandas.DataFrame
            """

            #Importing Yelp business data
            with open(os.path.join(dir, 'yelp_academic_dataset_business.json'), encoding='utf-8') as json_file:
                data = [json.loads(line) for line in json_file]    
                df_yelp_business = pd.DataFrame(data)
            
            #Importing Yelp review data
            with open(os.path.join(dir, 'yelp_academic_dataset_review.json'), encoding='utf-8') as json_file:
                data = [json.loads(line) for line in json_file]    
                df_yelp_review = pd.DataFrame(data)

        
            #Importing Yelp user data
            with open(os.path.join(dir, 'yelp_academic_dataset_user.json'), encoding='utf-8') as json_file:
                data = [json.loads(line) for line in json_file]    
                df_yelp_user = pd.DataFrame(data)
            
            #Importing Yelp checkin data
            with open(os.path.join(dir, 'yelp_academic_dataset_checkin.json'), encoding='utf-8') as json_file:
                data = [json.loads(line) for line in json_file]    
                df_yelp_checkin = pd.DataFrame(data)
                
            #Importing Yelp tip data
            with open(os.path.join(dir, 'yelp_academic_dataset_tip.json'), encoding='utf-8') as json_file:
                data = [json.loads(line) for line in json_file]    
                df_yelp_tip = pd.DataFrame(data) 
                
            return df_yelp_business, df_yelp_user, df_yelp_review, df_yelp_tip, df_yelp_checkin

        # processed_dir = osp.join(root, 'processed')

        untar_file_path = osp.join('/media/SSD4TB/users/hangyul/PEAGNN/datasets/Yelp', 'yelp_dataset')
        def drop_infrequent_concept_from_str(df, concept_name):
            concept_strs = [concept_str if concept_str != None else '' for concept_str in df[concept_name]]
            duplicated_concept = [concept_str.split(', ') for concept_str in concept_strs]
            duplicated_concept = list(itertools.chain.from_iterable(duplicated_concept))
            business_category_dict = Counter(duplicated_concept)
            del business_category_dict['']
            del business_category_dict['N/A']
            unique_concept = [k for k, v in business_category_dict.items() if
                            v >= 0.1 * np.max(list(business_category_dict.values()))]
            concept_strs = [
                ','.join([concept for concept in concept_str.split(', ') if concept in unique_concept])
                for concept_str in concept_strs
            ]
            df[concept_name] = concept_strs
            return df

        # print('Data frame not found in {}! Read from raw data!'.format(processed_dir))
        business, user, review, tip, checkin = parse_yelp(untar_file_path)

        print('Preprocessing...')
        # Extract business hours
        hours = []
        for hr in business['hours']:
            hours.append(hr) if hr != None else hours.append({})

        df_hours = (
            pd.DataFrame(hours)
                .fillna(False))

        # Replacing all times with True
        df_hours.where(df_hours == False, True, inplace=True)

        

        # Filter business categories > 1% of max value
        business = drop_infrequent_concept_from_str(business, 'categories')

        # Extract business attributes
        attributes = []
        for attr_list in business['attributes']:
            attr_dict = {}
            if attr_list != None:
                for a, b in attr_list.items():
                    if (b.lower() == 'true' or ''.join(re.findall(r"'(.*?)'", b)).lower() in (
                            'outdoor', 'yes', 'allages', '21plus', '19plus', '18plus', 'full_bar', 'beer_and_wine',
                            'yes_free', 'yes_corkage', 'free', 'paid', 'quiet', 'average', 'loud', 'very_loud',
                            'casual',
                            'formal', 'dressy')):
                        attr_dict[a.strip()] = True
                    elif (b.lower() in ('false', 'none') or ''.join(re.findall(r"'(.*?)'", b)).lower() in (
                            'no', 'none')):
                        attr_dict[a.strip()] = False
                    elif (b[0] != '{'):
                        attr_dict[a.strip()] = True
                    else:
                        for c in b.split(","):
                            attr_dict[a.strip()] = False
                            if (c == '{}'):
                                attr_dict[a.strip()] = False
                                break
                            elif (c.split(":")[1].strip().lower() == 'true'):
                                attr_dict[a.strip()] = True
                                break
            attributes.append([k for k, v in attr_dict.items() if v == True])

        business['attributes'] = [','.join(map(str, l)) for l in attributes]

        # Concating business df
        business_concat = [business.iloc[:, :-1], df_hours]
        business = pd.concat(business_concat, axis=1)

        # Compute friend counts
        user['friends_count'] = [len(f.split(",")) if f != 'None' else 0 for f in user['friends']]

        # Compute checkin counts
        checkin['checkin_count'] = [len(f.split(",")) if f != 'None' else 0 for f in checkin['date']]

        # Extract business checkin times
        checkin_years = []
        checkin_months = []
        checkin_time = []
        for checkin_list in checkin['date']:
            checkin_years_ar = []
            checkin_months_ar = []
            checkin_time_ar = []
            if checkin_list != '':
                for chk in checkin_list.split(","):
                    checkin_years_ar.append(chk.strip()[:4])
                    checkin_months_ar.append(chk.strip()[:7])

                    if int(chk.strip()[11:13]) in range(0, 4):
                        checkin_time_ar.append('00-03')
                    elif int(chk.strip()[11:13]) in range(3, 7):
                        checkin_time_ar.append('03-06')
                    elif int(chk.strip()[11:13]) in range(6, 10):
                        checkin_time_ar.append('06-09')
                    elif int(chk.strip()[11:13]) in range(9, 13):
                        checkin_time_ar.append('09-12')
                    elif int(chk.strip()[11:13]) in range(12, 16):
                        checkin_time_ar.append('12-15')
                    elif int(chk.strip()[11:13]) in range(15, 19):
                        checkin_time_ar.append('15-18')
                    elif int(chk.strip()[11:13]) in range(18, 22):
                        checkin_time_ar.append('18-21')
                    elif int(chk.strip()[11:13]) in range(21, 24):
                        checkin_time_ar.append('21-24')

            checkin_years.append(Counter(checkin_years_ar))
            checkin_months.append(Counter(checkin_months_ar))
            checkin_time.append(Counter(checkin_time_ar))

        df_checkin = (pd.concat([
            pd.DataFrame(checkin_years)
                .fillna('0').sort_index(axis=1),
            pd.DataFrame(checkin_months)
                .fillna('0').sort_index(axis=1),
            pd.DataFrame(checkin_time)
                .fillna('0').sort_index(axis=1)], axis=1))

        num_core = 10

        # Concating checkin df
        checkin_concat = [checkin, df_checkin]
        checkin = pd.concat(checkin_concat, axis=1)

        # Merging business and checkin
        business = pd.merge(business, checkin, on='business_id', how='left').fillna(0)

        # Select only relevant columns of review and tip
        review = review.iloc[:, [1, 2]]
        tip = tip.iloc[:, [0, 1]]

        # Concat review and tips
        reviewtip = pd.concat([review, tip], axis=0)

        # remove duplications
        business = business.drop_duplicates()
        user = user.drop_duplicates()
        reviewtip = reviewtip.drop_duplicates()

        if business.shape[0] != business.business_id.unique().shape[0] or user.shape[0] != \
                user.user_id.unique().shape[0]:
            raise ValueError('Duplicates in dfs.')

        #Filter only open business
        business = business[business.is_open == 1]

        # Compute the business counts for reviewtip
        bus_count = reviewtip['business_id'].value_counts()
        bus_count.name = 'bus_count'

        # Remove infrequent business in reviewtip
        reviewtip = reviewtip[reviewtip.join(bus_count, on='business_id').bus_count > (num_core + 40)]

        # Compute the user counts for reviewtip
        user_count = reviewtip['user_id'].value_counts()
        user_count.name = 'user_count'
        reviewtip = reviewtip.join(user_count, on='user_id')

        # Remove infrequent users in reviewtip
        reviewtip = reviewtip[
            (reviewtip.user_count > num_core) & (reviewtip.user_count <= (num_core + 10))]

        # Sync the business and user dataframe
        user = user[user.user_id.isin(reviewtip['user_id'].unique())]
        business = business[business.business_id.isin(reviewtip['business_id'].unique())]
        reviewtip = reviewtip[reviewtip.user_id.isin(user['user_id'].unique())]
        reviewtip = reviewtip[reviewtip.business_id.isin(business['business_id'].unique())]

        # Compute the updated business and user counts for reviewtip
        bus_count = reviewtip['business_id'].value_counts()
        user_count = reviewtip['user_id'].value_counts()
        bus_count.name = 'bus_count'
        user_count.name = 'user_count'
        reviewtip = reviewtip.iloc[:, [0, 1]].join(bus_count, on='business_id')
        reviewtip = reviewtip.join(user_count, on='user_id')

        def reindex_df(business, user, reviewtip):
            """
            reindex business, user, reviewtip in case there are some values missing or duplicates in between
            :param business: pd.DataFrame
            :param user: pd.DataFrame
            :param reviewtip: pd.DataFrame
            :return: same
            """
            print('Reindexing dataframes...')
            unique_uids = user.user_id.unique()
            unique_iids = business.business_id.unique()

            num_users = unique_uids.shape[0]
            num_bus = unique_iids.shape[0]

            raw_uids = np.array(unique_uids, dtype=object)
            raw_iids = np.array(unique_iids, dtype=object)

            uids = np.arange(num_users)
            iids = np.arange(num_bus)

            user['user_id'] = uids
            business['business_id'] = iids

            raw_uid2uid = {raw_uid: uid for raw_uid, uid in zip(raw_uids, uids)}
            raw_iid2iid = {raw_iid: iid for raw_iid, iid in zip(raw_iids, iids)}

            review_uids = np.array(reviewtip.user_id, dtype=object)
            review_iids = np.array(reviewtip.business_id, dtype=object)
            review_uids = [raw_uid2uid[review_uid] for review_uid in review_uids]
            review_iids = [raw_iid2iid[review_iid] for review_iid in review_iids]
            reviewtip['user_id'] = review_uids
            reviewtip['business_id'] = review_iids

            print('Reindex done!')

            return business, user, reviewtip

        # Reindex the bid and uid in case of missing values
        business, user, reviewtip = reindex_df(business, user, reviewtip)

        user2item_edge_index_train = np.zeros((2, 0))
        user2item_edge_index_test = np.zeros((2, 0))

        unique_uids = list(np.sort(reviewtip.user_id.unique()))
        num_uids = len(unique_uids)

        unique_iids = list(np.sort(reviewtip.business_id.unique()))
        num_iids = len(unique_iids)



        pbar = tqdm.tqdm(unique_uids, total=len(unique_uids))

        sorted_reviewtip = reviewtip.sort_values(['bus_count', 'user_count'])

        acc = 0
        uid2nid = {uid: i + acc for i, uid in enumerate(unique_uids)}

        acc += num_uids
        iid2nid = {iid: i + acc for i, iid in enumerate(unique_iids)}

        e2nid_dict = {'uid': uid2nid, 'iid': iid2nid }

        test_pos_unid_inid_map, neg_unid_inid_map = {}, {}

        for uid in pbar:
            pbar.set_description('Creating the edges for the user {}'.format(uid))
            uid_reviewtip = sorted_reviewtip[sorted_reviewtip.user_id == uid]
            uid_iids = uid_reviewtip.business_id.to_numpy()

            unid = e2nid_dict['uid'][uid]
            train_pos_uid_iids = list(uid_iids[:-1])  # Use leave one out setup
            train_pos_uid_inids = [e2nid_dict['iid'][iid] for iid in train_pos_uid_iids]
            test_pos_uid_iids = list(uid_iids[-1:])
            test_pos_uid_inids = [e2nid_dict['iid'][iid] for iid in test_pos_uid_iids]
            neg_uid_iids = list(set(unique_iids) - set(uid_iids))
            neg_uid_inids = [e2nid_dict['iid'][iid] for iid in neg_uid_iids]

            test_pos_unid_inid_map[unid] = test_pos_uid_inids
            neg_unid_inid_map[unid] = neg_uid_inids

            unid_user2item_edge_index_np = np.array(
                [[unid for _ in range(len(train_pos_uid_inids))], train_pos_uid_inids]
            )
            unid_user2item_edge_index_np_test = np.array(
                [[unid for _ in range(len(test_pos_uid_inids))], test_pos_uid_inids]
            )
            
            user2item_edge_index_train = np.hstack([user2item_edge_index_train, unid_user2item_edge_index_np])
            user2item_edge_index_test = np.hstack([user2item_edge_index_test, unid_user2item_edge_index_np_test])

        self.graph = torch_geometric.data.data.Data(num_nodes=num_uids+num_iids, edge_index=np.hstack([user2item_edge_index_train, user2item_edge_index_test]))
        self.split_edge = {
                            'train': {'edge': torch.Tensor(user2item_edge_index_train)},
                            'test': {'edge': torch.Tensor(user2item_edge_index_test)}
                            }


class DEDataset():
    def __init__(self, dataset, mask_ratio=0.05, use_weight=False, use_coalesce=False, use_degree=False,
                 use_val=False):
        self.data = Yelp()
        self.graph = self.data.graph
        self.split_edge = self.data.split_edge()
        self.mask_ratio = mask_ratio
        self.use_degree = use_degree
        self.use_weight = (use_weight and 'edge_weight' in self.graph)
        self.use_coalesce = use_coalesce
        self.use_val = use_val
        self.gtype = 'Homogeneous'

        if 'x' in self.graph:
            self.num_nodes, self.num_feature = self.graph['x'].shape
        else:
            self.num_nodes, self.num_feature = len(torch.unique(self.graph['edge_index'])), None

        if 'source_node' in self.split_edge['train']:
            self.directed = True
            self.train_edge = self.graph['edge_index'].t()
        else:
            self.directed = False
            self.train_edge = self.split_edge['train']['edge']

        if use_weight:
            self.train_weight = self.split_edge['train']['weight']
            if use_coalesce:
                train_edge_col, self.train_weight = coalesce(self.train_edge.t(), self.train_weight, self.num_nodes,
                                                             self.num_nodes)
                self.train_edge = train_edge_col.t()
            self.train_wmax = max(self.train_weight)
        else:
            self.train_weight = None
        # must put after coalesce
        self.len_train = self.train_edge.shape[0]

    def process(self, logger):
        logger.info(f'{self.data.meta_info}\nKeys: {self.graph.keys}')
        logger.info(
            f'node size {self.num_nodes}, feature dim {self.num_feature}, edge size {self.len_train} with mask ratio {self.mask_ratio}')
        logger.info(
            f'use_weight {self.use_weight}, use_coalesce {self.use_coalesce}, use_degree {self.use_degree}, use_val {self.use_val}')

        self.num_pos = int(self.len_train * self.mask_ratio)
        idx = np.random.permutation(self.len_train)
        # pos sample edges masked for training, observed edges for structural features
        self.pos_edge, obsrv_edge = self.train_edge[idx[:self.num_pos]], self.train_edge[idx[self.num_pos:]]
        val_edge = self.train_edge
        self.val_nodes = torch.unique(self.train_edge).tolist()

        if self.use_weight:
            pos_e_weight = self.train_weight[idx[:self.num_pos]]
            obsrv_e_weight = self.train_weight[idx[self.num_pos:]]
            val_e_weight = self.train_weight
        else:
            pos_e_weight = np.ones(self.num_pos, dtype=int)
            obsrv_e_weight = np.ones(self.len_train - self.num_pos, dtype=int)
            val_e_weight = np.ones(self.len_train, dtype=int)

        if self.use_val:
            # collab allows using valid edges for training
            obsrv_edge = torch.cat([obsrv_edge, self.split_edge['valid']['edge']])
            full_edge = torch.cat([self.train_edge, self.split_edge['valid']['edge']], dim=0)
            self.test_nodes = torch.unique(full_edge).tolist()
            if self.use_weight:
                obsrv_e_weight = torch.cat([self.train_weight[idx[self.num_pos:]], self.split_edge['valid']['weight']])
                full_e_weight = torch.cat([self.train_weight, self.split_edge['valid']['weight']], dim=0)
                if self.use_coalesce:
                    obsrv_edge_col, obsrv_e_weight = coalesce(obsrv_edge.t(), obsrv_e_weight, self.num_nodes,
                                                              self.num_nodes)
                    obsrv_edge = obsrv_edge_col.t()
                    full_edge_col, full_e_weight = coalesce(full_edge.t(), full_e_weight, self.num_nodes,
                                                            self.num_nodes)
                    full_edge = full_edge_col.t()
                self.full_wmax = max(full_e_weight)
            else:
                obsrv_e_weight = np.ones(obsrv_edge.shape[0], dtype=int)
                full_e_weight = np.ones(full_edge.shape[0], dtype=int)
        else:
            full_edge, full_e_weight = self.train_edge, self.train_weight
            self.test_nodes = self.val_nodes

        # load observed graph and save as a CSR sparse matrix
        max_obsrv_idx = torch.max(obsrv_edge).item()
        net_obsrv = csr_matrix((obsrv_e_weight, (obsrv_edge[:, 0].numpy(), obsrv_edge[:, 1].numpy())),
                               shape=(max_obsrv_idx + 1, max_obsrv_idx + 1))
        G_obsrv = net_obsrv + net_obsrv.T
        assert sum(G_obsrv.diagonal()) == 0

        # subgraph for training(5 % edges, pos edges)
        max_pos_idx = torch.max(self.pos_edge).item()
        net_pos = csr_matrix((pos_e_weight, (self.pos_edge[:, 0].numpy(), self.pos_edge[:, 1].numpy())),
                             shape=(max_pos_idx + 1, max_pos_idx + 1))
        G_pos = net_pos + net_pos.T
        assert sum(G_pos.diagonal()) == 0

        max_val_idx = torch.max(val_edge).item()
        net_val = csr_matrix((val_e_weight, (val_edge[:, 0].numpy(), val_edge[:, 1].numpy())),
                             shape=(max_val_idx + 1, max_val_idx + 1))
        G_val = net_val + net_val.T
        assert sum(G_val.diagonal()) == 0

        if self.use_val:
            max_full_idx = torch.max(full_edge).item()
            net_full = csr_matrix((full_e_weight, (full_edge[:, 0].numpy(), full_edge[:, 1].numpy())),
                                  shape=(max_full_idx + 1, max_full_idx + 1))
            G_full = net_full + net_full.transpose()
            assert sum(G_full.diagonal()) == 0
        else:
            G_full = G_val

        self.degree = np.expand_dims(np.log(G_full.getnnz(axis=1) + 1), 1).astype(
            np.float32) if self.use_degree else None

        # sparsity of graph
        logger.info(f'Sparsity of loaded graph {G_obsrv.getnnz() / (max_obsrv_idx + 1) ** 2}')
        # statistic of graph
        logger.info(
            f'Observed subgraph with {np.sum(G_obsrv.getnnz(axis=1) > 0)} nodes and {int(G_obsrv.nnz / 2)} edges;')
        logger.info(f'Training subgraph with {np.sum(G_pos.getnnz(axis=1) > 0)} nodes and {int(G_pos.nnz / 2)} edges.')

        self.data, self.graph = None, None

        return {'pos': G_pos, 'train': G_obsrv, 'val': G_val, 'test': G_full}


class DE_Hetro_Dataset():
    def __init__(self, dataset, relation, mask_ratio=0.05):
        self.data = torch.load(f'./dataset/{dataset}_{relation}.pl')
        self.split_edge = self.data['split_edge']
        self.node_type = list(self.data['num_nodes_dict'])
        self.mask_ratio = mask_ratio
        rel_key = ('author', 'writes', 'paper') if relation == 'cite' else ('paper', 'cites', 'paper')
        self.obsrv_edge = self.data['edge_index'][rel_key]
        self.split_edge = self.data['split_edge']
        self.gtype = 'Heterogeneous' if relation == 'cite' else 'Homogeneous'

        if 'x' in self.data:
            self.num_nodes, self.num_feature = self.data['x'].shape
        else:
            self.num_nodes, self.num_feature = self.obsrv_edge.unique().size(0), None

        if 'source_node' in self.split_edge['train']:
            self.directed = True
            self.train_edge = self.graph['edge_index'].t()
        else:
            self.directed = False
            self.train_edge = self.split_edge['train']['edge']

        self.len_train = self.train_edge.shape[0]

    def process(self, logger):
        logger.info(
            f'node size {self.num_nodes}, feature dim {self.num_feature}, edge size {self.len_train} with mask ratio {self.mask_ratio}')

        self.num_pos = int(self.len_train * self.mask_ratio)
        idx = np.random.permutation(self.len_train)
        # pos sample edges masked for training, observed edges for structural features
        self.pos_edge, obsrv_edge = self.train_edge[idx[:self.num_pos]], torch.cat(
            [self.train_edge[idx[self.num_pos:]], self.obsrv_edge])
        val_edge = torch.cat([self.train_edge, self.obsrv_edge])
        len_redge = len(self.obsrv_edge)

        pos_e_weight = np.ones(self.num_pos, dtype=int)
        obsrv_e_weight = np.ones(self.len_train - self.num_pos + len_redge, dtype=int)
        val_e_weight = np.ones(self.len_train + len_redge, dtype=int)

        # load observed graph and save as a CSR sparse matrix
        max_obsrv_idx = torch.max(obsrv_edge).item()
        net_obsrv = csr_matrix((obsrv_e_weight, (obsrv_edge[:, 0].numpy(), obsrv_edge[:, 1].numpy())),
                               shape=(max_obsrv_idx + 1, max_obsrv_idx + 1))
        G_obsrv = net_obsrv + net_obsrv.T
        assert sum(G_obsrv.diagonal()) == 0

        # subgraph for training(5 % edges, pos edges)
        max_pos_idx = torch.max(self.pos_edge).item()
        net_pos = csr_matrix((pos_e_weight, (self.pos_edge[:, 0].numpy(), self.pos_edge[:, 1].numpy())),
                             shape=(max_pos_idx + 1, max_pos_idx + 1))
        G_pos = net_pos + net_pos.T
        assert sum(G_pos.diagonal()) == 0

        max_val_idx = torch.max(val_edge).item()
        net_val = csr_matrix((val_e_weight, (val_edge[:, 0].numpy(), val_edge[:, 1].numpy())),
                             shape=(max_val_idx + 1, max_val_idx + 1))
        G_val = net_val + net_val.T
        assert sum(G_val.diagonal()) == 0

        G_full = G_val
        # sparsity of graph
        logger.info(f'Sparsity of loaded graph {G_obsrv.getnnz() / (max_obsrv_idx + 1) ** 2}')
        # statistic of graph
        logger.info(
            f'Observed subgraph with {np.sum(G_obsrv.getnnz(axis=1) > 0)} nodes and {int(G_obsrv.nnz / 2)} edges;')
        logger.info(f'Training subgraph with {np.sum(G_pos.getnnz(axis=1) > 0)} nodes and {int(G_pos.nnz / 2)} edges.')

        self.data = None
        return {'pos': G_pos, 'train': G_obsrv, 'val': G_val, 'test': G_full}


class DE_Hyper_Dataset():
    def __init__(self, dataset, mask_ratio=0.6):
        self.data = torch.load(f'./dataset/{dataset}.pl')
        self.obsrv_edge = torch.from_numpy(self.data['edge_index'])
        self.num_tup = len(self.data['triplets'])
        self.mask_ratio = mask_ratio
        self.split_edge = self.data['triplets']
        self.gtype = 'Hypergraph'

        if 'x' in self.data:
            self.num_nodes, self.num_feature = self.data['x'].shape
        else:
            self.num_nodes, self.num_feature = self.obsrv_edge.unique().size(0), None

    def get_edge_split(self, ratio, k=1000, seed=2021):
        np.random.seed(seed)
        tuples = torch.from_numpy(self.data['triplets'])
        idx = np.random.permutation(self.num_tup)
        num_train = int(ratio * self.num_tup)
        split_idx = {'train': {'hedge': tuples[idx[:num_train]]}}
        val_idx, test_idx = np.split(idx[num_train:], 2)
        split_idx['valid'], split_idx['test'] = {'hedge': tuples[val_idx]}, {'hedge': tuples[test_idx]}
        node_neg = torch.randint(torch.max(tuples), (len(val_idx), k))
        split_idx['valid']['hedge_neg'] = torch.cat(
            [split_idx['valid']['hedge'][:, :2].repeat(1, k).view(-1, 2).t(), node_neg.view(1, -1)]).t()
        split_idx['test']['hedge_neg'] = torch.cat(
            [split_idx['test']['hedge'][:, :2].repeat(1, k).view(-1, 2).t(), node_neg.view(1, -1)]).t()
        return split_idx

    def process(self, logger):
        logger.info(
            f'node size {self.num_nodes}, feature dim {self.num_feature}, edge size {self.num_tup} with mask ratio {self.mask_ratio}')
        obsrv_edge = self.obsrv_edge

        # load observed graph and save as a CSR sparse matrix
        max_obsrv_idx = torch.max(obsrv_edge).item()
        obsrv_e_weight = np.ones(len(obsrv_edge), dtype=int)
        net_obsrv = csr_matrix((obsrv_e_weight, (obsrv_edge[:, 0].numpy(), obsrv_edge[:, 1].numpy())),
                               shape=(max_obsrv_idx + 1, max_obsrv_idx + 1))
        G_enc = net_obsrv + net_obsrv.T
        assert sum(G_enc.diagonal()) == 0

        # sparsity of graph
        logger.info(f'Sparsity of loaded graph {G_enc.getnnz() / (max_obsrv_idx + 1) ** 2}')
        # statistic of graph
        logger.info(f'Observed subgraph with {np.sum(G_enc.getnnz(axis=1) > 0)} nodes and {int(G_enc.nnz / 2)} edges;')

        return G_enc


def gen_dataset(dataset, graphs, args, bsize=10000):
    G_val, G_full = graphs['val'], graphs['test']

    keep_neg = False if 'ppa' not in args.dataset else True

    test_pos_edge, test_neg_edge = get_pos_neg_edges('test', dataset.split_edge, ratio=args.test_ratio,
                                                     keep_neg=keep_neg)
    val_pos_edge, val_neg_edge = get_pos_neg_edges('valid', dataset.split_edge, ratio=args.valid_ratio,
                                                   keep_neg=keep_neg)

    inf_set = {'test': {}, 'val': {}}

    if args.metric == 'mrr':
        inf_set['test']['E'] = torch.cat([test_pos_edge, test_neg_edge], dim=1).t()
        inf_set['val']['E'] = torch.cat([val_pos_edge, val_neg_edge], dim=1).t()
        inf_set['test']['num_pos'], inf_set['val']['num_pos'] = test_pos_edge.shape[1], val_pos_edge.shape[1]
        inf_set['test']['num_neg'], inf_set['val']['num_neg'] = test_neg_edge.shape[1] // inf_set['test']['num_pos'], \
                                                                val_neg_edge.shape[1] // inf_set['val']['num_pos']
    elif 'Hit' in args.metric:
        inf_set['test']['E'] = torch.cat([test_neg_edge, test_pos_edge], dim=1).t()
        inf_set['val']['E'] = torch.cat([val_neg_edge, val_pos_edge], dim=1).t()
        inf_set['test']['num_pos'], inf_set['val']['num_pos'] = test_pos_edge.shape[1], val_pos_edge.shape[1]
        inf_set['test']['num_neg'], inf_set['val']['num_neg'] = test_neg_edge.shape[1], val_neg_edge.shape[1]
    else:
        raise NotImplementedError

    if args.use_val:
        val_dict = np_sampling({}, G_val.indptr, G_val.indices, bsize=bsize,
                               target=torch.unique(inf_set['val']['E']).tolist(), num_walks=args.num_walk,
                               num_steps=args.num_step - 1)
        test_dict = np_sampling({}, G_full.indptr, G_full.indices, bsize=bsize,
                                target=torch.unique(inf_set['test']['E']).tolist(), num_walks=args.num_walk,
                                num_steps=args.num_step - 1)
    else:
        val_dict = test_dict = np_sampling({}, G_val.indptr, G_val.indices, bsize=bsize,
                                           target=torch.unique(
                                               torch.cat([inf_set['val']['E'], inf_set['test']['E']])).tolist(),
                                           num_walks=args.num_walk, num_steps=args.num_step - 1)

    if not args.use_feature:
        if args.use_degree:
            inf_set['X'] = torch.from_numpy(dataset.degree)
        elif args.use_htype:
            inf_set['X'] = dataset.node_map
        else:
            inf_set['X'] = None
    else:
        inf_set['X'] = dataset.graph['x']
        args.x_dim = inf_set['X'].shape[-1]

    args.w_max = dataset.train_wmax if args.use_weight else None

    return test_dict, val_dict, inf_set


def gen_dataset_hyper(dataset, G_enc, args, bsize=10000):
    test_pos_edge, test_neg_edge = get_pos_neg_edges('test', dataset.split_edge, ratio=args.test_ratio)
    val_pos_edge, val_neg_edge = get_pos_neg_edges('valid', dataset.split_edge, ratio=args.valid_ratio)

    inf_set = {'test': {}, 'val': {}}

    if args.metric == 'mrr':
        inf_set['test']['E'] = torch.cat([test_pos_edge, test_neg_edge])
        inf_set['val']['E'] = torch.cat([val_pos_edge, val_neg_edge])
        inf_set['test']['num_pos'], inf_set['val']['num_pos'] = test_pos_edge.shape[0], val_pos_edge.shape[0]
        inf_set['test']['num_neg'], inf_set['val']['num_neg'] = test_neg_edge.shape[0] // inf_set['test']['num_pos'], \
                                                                val_neg_edge.shape[0] // inf_set['val']['num_pos']
    else:
        raise NotImplementedError

    inf_dict = np_sampling({}, G_enc.indptr, G_enc.indices,
                           bsize=bsize,
                           target=torch.unique(torch.cat([inf_set['val']['E'], inf_set['test']['E']])).tolist(),
                           num_walks=args.num_walk,
                           num_steps=args.num_step - 1)

    if not args.use_feature:
        inf_set['X'] = None
    else:
        inf_set['X'] = dataset.graph['x']
        args.x_dim = inf_set['X'].shape[-1]

    return inf_dict, inf_set
