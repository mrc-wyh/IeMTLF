import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import math
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import os, sys
os.chdir(sys.path[0])


class GeoGCNLayer(nn.Module):
    def __init__(self,
                g,
                args,
                device='cuda'
                ):
        super(GeoGCNLayer, self).__init__()
        self.g = g
        self.device = device
        self.act = nn.LeakyReLU(0.2)
        self.is_att = args.is_att #地理空间影响和时序转移影响融合是否采用注意力方式
        self.is_sgc = args.is_sgc #是否采用SGC方式，即去掉非线性变化处理
        # self.geo_w = args.geo_w
        # self.tran_w = args.tran_w
        # self.cat_w = args.cat_w
        self.is_lightgcn = args.is_lightgcn
        if self.is_att:
            self.attn_fuse = SemanticAttention(args.hidden_dim, args.hidden_dim*4)
        if not args.is_lightgcn:
            self.feat_tran = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
            nn.init.xavier_uniform_(self.feat_tran.weight, gain=1.414)
        
    def forward(self, feat):
        funcs = {}#message and reduce functions dict
        feat_t = feat if self.is_lightgcn else self.feat_tran(feat)
        self.g.ndata['f'] = feat_t
        for srctype, etype, dsttype in self.g.canonical_etypes:
            if etype == 'geo':
                funcs[etype] = (fn.copy_u('f', 'm'), fn.mean('m', 'geo'))
            elif etype == 'cat':
                funcs[etype] = (fn.copy_u('f', 'm'), fn.mean('m', 'cat'))
            else:
                funcs[etype] = (fn.u_mul_e('f', 'w', 'm'), fn.sum('m', 'trans'))
        # for srctype, etype, dsttype in self.g.canonical_etypes:
        #     if etype == 'geo':
        #         if self.geo_w == 0 and not self.is_att:
        #             continue
        #         else:
        #             funcs[etype] = (fn.copy_u('f', 'm'), fn.mean('m', 'geo'))
        #     elif etype == 'trans':
        #         if self.tran_w == 0 and not self.is_att:
        #             continue
        #         else:
        #             funcs[etype] = (fn.u_mul_e('f', 'w', 'm'), fn.sum('m', 'trans'))
        #     else:
        #         if self.cat_w == 0 and not self.is_att:
        #             continue
        #         else:
        #             funcs[etype] = (fn.copy_u('f', 'm'), fn.mean('m', 'cat'))
                    
        self.g.multi_update_all(funcs, 'sum')
        if self.is_att: #采用注意力融合
            geo = self.g.ndata['geo'].unsqueeze(1)
            trans = self.g.ndata['trans'].unsqueeze(1)
            cat = self.g.ndata['cat'].unsqueeze(1)
            # z = torch.cat([geo, trans], 1)
            z = torch.cat([geo, trans, cat], 1)
            feat = self.attn_fuse(z)
        # else:#采用加权方式（人工设置相应权重系数）
        #     if self.geo_w == 0:
        #         feat = self.g.ndata['trans']
        #     elif self.tran_w == 0:
        #         feat = self.g.ndata['geo']
        #     else:
        #         feat = self.geo_w * self.g.ndata['geo'] + self.tran_w * self.g.ndata['trans']
        return feat if self.is_sgc else self.act(feat)

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1) M：元路径数量
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        # print(beta)
        beta = beta.expand((z.shape[0],) + beta.shape) 
        return (beta * z).sum(1)                       
                                              
class GeoGCN(nn.Module):
    def __init__(self,
                g,
                tran_e_w,
                args,
                device='cuda'
                ):
        super(GeoGCN, self).__init__()
        # self.tran_w = torch.tensor(tran_weight).to(device)
        g = g.int()
        g = dgl.remove_self_loop(g, etype='geo')
        g = dgl.add_self_loop(g, etype='geo')
        self.g = g.to(device)
        self.g.edges['trans'].data['w'] = torch.tensor(tran_e_w).float().to(device)
        self.num_layer = args.GeoGCN_layer_num
        self.dropout = args.gcn_drop
        self.device = device
        self.act = nn.LeakyReLU(0.2)
        # self.is_space2vec = args.is_space2vec
        # if args.is_space2vec:
        #     self.feat_tran = nn.Linear(args.frequency_num * 4, args.hidden_dim, bias=False)
        #     nn.init.xavier_uniform_(self.l_l.weight, gain=1.414)
            
        self.gcn = nn.ModuleList()
        for i in range(self.num_layer):
            self.gcn.append(
            GeoGCNLayer(self.g, args, device)
        )
            
    def forward(self, feat):
        for i in range(self.num_layer - 1):
            feat = self.gcn[i](feat)
        if self.num_layer > 1:
            feat = F.dropout(feat, self.dropout)
        feat = self.gcn[-1](feat)
        return feat

class SlotEncoding(nn.Module):
    "Position Encoding module"
    def __init__(self, dim_model, max_len=100, device='cuda'):
        super(SlotEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2, dtype=torch.float) *
                             -(np.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(device)
        self.register_buffer('pe', pe)  # Not a parameter but should be in state_dict
    
    def forward(self, pos):
        return torch.index_select(self.pe, 1, pos).squeeze(0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MLP(nn.Module):
    def __init__(self, inpu_dim, hidden, out_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(inpu_dim, hidden)
        self.w_2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2)
        # nn.init.xavier_uniform_(self.w_1.weight, gain=1.414)
        # nn.init.xavier_uniform_(self.w_2.weight, gain=1.414)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.w_1.bias)
        nn.init.zeros_(self.w_2.bias)
        nn.init.uniform_(self.w_1.weight, -initrange, initrange)
        nn.init.uniform_(self.w_2.weight, -initrange, initrange)
    
    def forward(self, x, time_given):
        tmp = self.dropout(self.act(self.w_1(x)))
        out = self.w_2(tmp * time_given)
        return out

class SeqPred(nn.Module):
    def __init__(self,
                cat_num,
                loc_num,
                args,
                slotencoding,
                device='cuda'
                ):
        super(SeqPred, self).__init__()
        self.dim = args.hidden_dim * 2 + args.time_dim * 3 + 12 * 2
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.dim, args.enc_drop)
        encoder_layers = TransformerEncoderLayer(self.dim, args.enc_nhead, args.enc_ffn_hdim, args.enc_drop)
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.enc_layer_num)
        self.decoder_cat = MLP(self.dim, args.time_dim, cat_num) if args.dec_time else nn.Linear(self.dim, cat_num)
        # self.decoder_cat = nn.Linear(self.dim, cat_num)#解码器->预测语义类别
        self.decoder_loc = nn.Linear(self.dim, loc_num)#解码器->预测具体位置
        self.time_given = slotencoding
        self.dec_time = args.dec_time
        self.init_weights()
        self.device = device
        
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder_loc.bias)
        nn.init.uniform_(self.decoder_loc.weight, -initrange, initrange)
        if not self.dec_time:
            nn.init.zeros_(self.decoder_cat.bias)
            nn.init.uniform_(self.decoder_cat.weight, -initrange, initrange)
        
        
    def forward(self, src, key_pad_mask, day_mode, week_mode, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        
        src = src * math.sqrt(self.dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask, key_pad_mask.transpose(0, 1)) #shape * batch *dim
        loc_out = self.decoder_loc(output)
        # correct = []
        # values = trans_matrix.data
        # indices = np.vstack((trans_matrix.row, trans_matrix.col))
        # i = torch.LongTensor(indices)
        # v = torch.FloatTensor(values)
        # shape = trans_matrix.shape
        # trans_matrix_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(self.device)
        # for i in range(len(prior_loc)):
        #     tmp = []
        #     for j in range(len(prior_loc[i])):
        #         tmp.append(trans_matrix_tensor[prior_loc[i][j]].to_dense())
        #     correct.append(torch.stack(tmp))
        # correct = pad_sequence(correct)
        # loc_out = loc_out + correct  
        if self.dec_time:
            day_mode_emb = []
            week_mode_emb = []
            for i in range(len(day_mode)):
                day_mode_emb.append(self.time_given(day_mode[i]))
                week_mode_emb.append(self.time_given(week_mode[i]))
            day_mode_emb = pad_sequence(day_mode_emb, batch_first=False, padding_value=0.5)
            week_mode_emb = pad_sequence(week_mode_emb, batch_first=False, padding_value=0.5)
            
            time_given_emb = day_mode_emb + week_mode_emb
            cat_out = self.decoder_cat(output, time_given_emb)
        else:
            cat_out = self.decoder_cat(output)
        return loc_out, cat_out

'''
class FFN(nn.Module):
    def __init__(self, inpu_dim, hidden, out_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(inpu_dim, hidden)
        self.w_2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.w_1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.w_2.weight, gain=1.414)

    def forward(self, x):
        return self.w_2(self.dropout(self.act(self.w_1(x))))
        # return self.w_2(self.dropout(torch.relu(self.w_1(x))))
'''

class Gen_Coaction(nn.Module):
    def __init__(self, args, orders=2):
        super(Gen_Coaction, self).__init__()
        self.weight_emb_w = args.co_action_w
        self.orders = orders

    def forward(self, input, mlp):
        weight_orders = []
        for i in range(self.orders):
            weight= []
            idx = 0 
            for w in self.weight_emb_w:
                weight.append(torch.reshape(mlp[:, idx:idx+w[0]*w[1]], [-1, w[0], w[1]]))
                idx += w[0] * w[1]
            weight_orders.append(weight)

        out_seq = []
        hh = []
        for i in range(self.orders):
            hh.append(input ** (i+1))
        for i, h in enumerate(hh):
            weight = weight_orders[0]
            h_order = []
            for j, w in enumerate(weight):
                h = torch.matmul(h, w)
                if j != len(weight)-1:
                    h = torch.tanh(h)
                h_order.append(h)
            out_seq.append(torch.concat(h_order, 2))
        out = torch.sum(torch.concat(out_seq, 1), 1)
        return out

'''    
class Interaction(nn.Module):#MLP方式的交互
    def __init__(self, dim1, dim2, out_dim):
        super(Interaction, self).__init__()
        in_dim = dim1 + dim2
        self.mul_inter = nn.Linear(in_dim, out_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.mul_inter.weight, gain=1.414)


    def forward(self, emb1, emb2):
        x = self.mul_inter(torch.cat((emb1, emb2), 1))
        x = self.leaky_relu(x)
        return x
'''

class DatasetPrePare(Dataset):
    def __init__(self, forward, label, user):
        self.forward = forward
        self.label = label
        self.user = user

    def __len__(self):
        assert len(self.forward) == len(self.label) == len(self.user)
        return len(self.forward)

    def __getitem__(self, index):
        return (self.forward[index], self.label[index], self.user[index])

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None
        self.best_epoch_val_loss = 0
        
    def step(self, score, loss, user_model, cat_model, loc_model, gcn_model, enc_model, user_mlp_model, cat_mlp_model, loc_input_model, epoch, result_dir):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_epoch_val_loss = loss
            self.save_checkpoint(user_model, cat_model, loc_model, gcn_model, enc_model, user_mlp_model, cat_mlp_model, loc_input_model, result_dir)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_epoch_val_loss = loss
            self.save_checkpoint(user_model, cat_model, loc_model, gcn_model, enc_model, user_mlp_model, cat_mlp_model, loc_input_model, result_dir)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, user_model, cat_model, loc_model, gcn_model, enc_model, user_mlp_model, cat_mlp_model, loc_input_model, result_dir):
        # Saves model when validation loss decrease.
        state_dict = {
            'user_emb_model_state_dict': user_model.state_dict(),
            'cat_emb_model_state_dict': cat_model.state_dict(),
            'loc_emb_model_state_dict': loc_model.state_dict(),
            'user_mlp_model_state_dict': user_mlp_model.state_dict(),
            'cat_mlp_model_state_dict': cat_mlp_model.state_dict(),
            'loc_input_model_state_dict': loc_input_model.state_dict(),
            'geogcn_model_state_dict': gcn_model.state_dict(),
            'transformer_encoder_model_state_dict': enc_model.state_dict()
            } 
        best_result = os.path.join(result_dir, 'checkpoint.pt')    
        torch.save(state_dict, best_result) 

'''       
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None
        self.best_epoch_val_loss = 0
        
    def step(self, score, loss, user_model, loc_model, gcn_model, enc_model, epoch, result_dir):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_epoch_val_loss = loss
            self.save_checkpoint(user_model, loc_model, gcn_model, enc_model, result_dir)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_epoch_val_loss = loss
            self.save_checkpoint(user_model, loc_model, gcn_model, enc_model, result_dir)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, user_model, loc_model, gcn_model, enc_model, result_dir):
        # Saves model when validation loss decrease.
        state_dict = {
            'user_emb_model_state_dict': user_model.state_dict(),
            # 'cat_emb_model_state_dict': cat_model.state_dict(),
            'loc_emb_model_state_dict': loc_model.state_dict(),
            'geogcn_model_state_dict': gcn_model.state_dict(),
            'transformer_encoder_model_state_dict': enc_model.state_dict()
            } 
        best_result = os.path.join(result_dir, 'checkpoint.pt')    
        torch.save(state_dict, best_result)
''' 