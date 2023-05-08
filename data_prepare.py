import pandas as pd
import numpy as np
# import torch
# import random as rd
import scipy.sparse as sp
import dgl
import pickle
import os, sys
os.chdir(sys.path[0])

class Data(object):
    def __init__(self, args):
        dataset = args.dataset
        
        loc_file = dataset + '/loc.csv'
        if args.dist == 1000:
            geo_edge_file = dataset + '/geo_edge_1000.csv'
        elif args.dist == 500:
            geo_edge_file = dataset + '/geo_edge_500.csv'
        elif args.dist == 1500:
            geo_edge_file = dataset + '/geo_edge_1500.csv'
        else:
            geo_edge_file = dataset + '/geo_edge_2000.csv'
        tran_edge_file = dataset + '/tran_edge.csv'
        cat_edge_file = dataset + '/cat_edge.csv'
        user_id_file = dataset + '/user_id.csv'
        train_forward_file = dataset + '/train_forward.pickle'
        train_labels_file = dataset + '/train_labels.pickle'
        train_user_file = dataset + '/train_user.pickle'
        valid_forward_file = dataset + '/valid_forward.pickle'
        valid_labels_file = dataset + '/valid_labels.pickle'
        valid_user_file = dataset + '/valid_user.pickle'
        test_forward_file = dataset + '/test_forward.pickle'
        test_lables_file = dataset + '/test_labels.pickle'
        test_user_file = dataset + '/test_user.pickle'
        
        loc = pd.read_csv(loc_file, names=['loc_ID', 'loc_cat_new_name', 'cat_id', 'loc_catin_id', 'latitude', 'longitude', 'loc_new_ID'], sep=',', header=0)
        user = pd.read_csv(user_id_file, names=['old_id', 'new_id'], sep=',', header=0)
        geo_edge = pd.read_csv(geo_edge_file, names=['src', 'dst'], sep=',', header=0)
        tran_edge = pd.read_csv(tran_edge_file, names=['src', 'dst', 'freq', 'weight'], sep=',', header=0)
        cat_edge = pd.read_csv(cat_edge_file, names=['src', 'dst'], sep=',', header=0)
        self.cat_num = max(loc['cat_id']) + 1
        self.loc_num = max(loc['loc_new_ID']) + 1
        self.user_num = max(user['new_id']) + 1
        loc_cat = loc[['loc_new_ID', 'cat_id']]
        loc_cat.sort_values(by='loc_new_ID', ascending=True, inplace=True)
        loc_map_cat = list(loc_cat['cat_id'])
        # loc_cat_g = loc_cat.groupby(by='cat_id')
        # for i, loc_g in loc_cat_g:
        #     l = list(loc_g['loc_new_ID'])
        #     cat_has_loc.append(l)
        self.loc_cat = loc_map_cat
        self.loc_g, self.tran_edge_weight = self.build_graph(args, geo_edge, tran_edge, cat_edge)        
        # self.trans_matrix = self.tran_matrix(tran_edge)        
        # self.loc_g, self.tran_edge_weight = self.build_graph(geo_edge, tran_edge)
        print(self.cat_num, self.loc_num, self.user_num)
        
        train_forward = open(train_forward_file,'rb')
        self.train_forward = pickle.load(train_forward)
        train_labels = open(train_labels_file,'rb')
        self.train_labels = pickle.load(train_labels)
        train_user = open(train_user_file,'rb')
        self.train_user = pickle.load(train_user)
        valid_forward = open(valid_forward_file,'rb')
        self.valid_forward = pickle.load(valid_forward)
        valid_labels = open(valid_labels_file,'rb')
        self.valid_labels = pickle.load(valid_labels)
        valid_user = open(valid_user_file,'rb')
        self.valid_user = pickle.load(valid_user)
        test_forward = open(test_forward_file,'rb')
        self.test_forward = pickle.load(test_forward)
        test_labels = open(test_lables_file,'rb')
        self.test_labels = pickle.load(test_labels)
        test_user = open(test_user_file,'rb')
        self.test_user = pickle.load(test_user)
        print('train traj num:', len(self.train_forward))
        print('valid traj num:', len(self.valid_forward))
        print('test traj num:', len(self.test_forward))

    # def build_graph(self, args, geo_edge, tran_edge, cat_edge):
    #     geo = np.array(geo_edge)
    #     geo_e = [tuple(geo[i]) for i in range(len(geo))]
    #     cat = np.array(cat_edge)
    #     cat_e = [tuple(cat[i]) for i in range(len(cat))]
    #     tran = np.array(tran_edge[['src', 'dst']])
    #     tran_e_w = np.array(tran_edge['weight'])
    #     tran_e = [tuple(tran[i]) for i in range(len(tran))]
    #     g_list = []
    #     geo_g = dgl.graph(geo_e)
    #     tran_g = dgl.graph(tran_e)
    #     g_list.append(geo_g)
    #     g_list.append(tran_g)
    #     if not args.base and args.cp4:
    #         cat_g = dgl.graph(cat_e)
    #         g_list.append(cat_g)
    #     return g_list, tran_e_w
       
    def build_graph(self, args, geo_edge, tran_edge, cat_edge):
        geo = np.array(geo_edge)
        geo_e = [tuple(geo[i]) for i in range(len(geo))]
        cat = np.array(cat_edge)
        cat_e = [tuple(cat[i]) for i in range(len(cat))]
        tran = np.array(tran_edge[['src', 'dst']])
        tran_e_w = np.array(tran_edge['weight'])
        tran_e = [tuple(tran[i]) for i in range(len(tran))]
        if not args.base and args.cp4:
            data_dict = {
                ('loc', 'geo', 'loc'): geo_e,
                ('loc', 'cat', 'loc'): cat_e,
                ('loc', 'trans', 'loc'): tran_e
            }
        else:
            data_dict = {
            ('loc', 'geo', 'loc'): geo_e,
            ('loc', 'trans', 'loc'): tran_e
            }
        return dgl.heterograph(data_dict), tran_e_w
    
    def tran_matrix(self, tran_edge):
        trans = tran_edge[['src', 'dst', 'freq']]
        trans = trans.sort_values(by='src').replace(True)
        trans.index = range(len(trans))
        gtedge_g = trans.groupby(by='src')
        trans_prob = pd.DataFrame()
        for src, e in gtedge_g:
            e.index = range(len(e))
            total_freq = e['freq'].sum()
            e['weight'] = e['freq'] / total_freq
            trans_prob = pd.concat((trans_prob, e))
        trans_prob = trans_prob.sort_values(by=['src', 'dst']).replace(True)
        trans_prob = trans_prob[['src', 'dst', 'weight']]
        trans_prob.index = range(len(trans_prob))
        row = np.array(trans_prob['src'])
        col = np.array(trans_prob['dst'])
        data = np.array(trans_prob['weight'])
        
        return sp.coo_matrix((data, (row, col)), shape=(self.loc_num, self.loc_num), dtype=np.float)