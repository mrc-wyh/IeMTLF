#version #2023.03.01
import datetime, os
import argparse, logging
import numpy as np
from numpy import random
import torch
import torch.nn as nn
import dgl
from data_prepare import Data
from model import SeqPred, SlotEncoding, GeoGCN, DatasetPrePare, EarlyStopping, Gen_Coaction
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from time import time
from pathlib import Path
import re
import glob

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)

def increment_path(path, exist_ok=False, sep='', mkdir=True):
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir_ = path if path.suffix == '' else path.parent  # directory
    if not dir_.exists() and mkdir:
        dir_.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def batch_seq_emb(args, data_b, max, cat_emb, loc_emb, pos_emb, user_emb_model, user_mlp_model, cat_mlp_model, loc_input_model, coaction, device):
    traj_n = len(data_b) #batch值
    batch_input_emb = []
    batch_loc_labels = []
    batch_cat_labels = []
    labels = []
    lengh = []
    # prior_labels = []
    prior_loc = []
    src_lengths = []
    day_mode = []
    week_mode = []
    for i in range(traj_n):        
        if len(data_b[i][0]) <= max:
            traj_forward = np.array(data_b[i][0])
            traj_labels = np.array(data_b[i][1])    
        else:
            traj_forward = np.array(data_b[i][0][-max:])
            traj_labels = np.array(data_b[i][1][-max:])
        loc  = traj_forward[:, 0]
        loc_labels = traj_labels[:, 0]
        timeslot = traj_forward[:, 1]
        day_of_week = traj_forward[:, 2]
        day_mode.append(torch.tensor(traj_labels[:, 1]).to(device))
        week_mode.append(torch.tensor(traj_labels[:, 2]).to(device))
        cat = traj_forward[:, 3]
        cat_labels = traj_labels[:, 3]
        loc_s_emb = loc_emb[loc] #shape:traj_len * hidden_dim
        cat_s_emb = cat_emb[cat] #shape:traj_len * hidden_dim
        timeslot_emb = pos_emb(torch.tensor(timeslot).to(device))
        dw_emb = pos_emb(torch.tensor(day_of_week).to(device))
        user_emb = user_emb_model(torch.tensor(data_b[i][2]).to(device)).unsqueeze(0)
        user_mlp = user_mlp_model(torch.tensor(data_b[i][2]).to(device)).unsqueeze(0)
        cat_mlp = cat_mlp_model(torch.tensor(cat).to(device))
        loc_input = loc_input_model(torch.tensor(loc).to(device))

        user_emb = user_emb.expand(len(loc_s_emb), -1)
        user_mlp = user_mlp.expand(len(loc_s_emb), -1)
        
        if args.interaction:
            time_emb = timeslot_emb + dw_emb
            cat_time = cat_s_emb * time_emb#显式交互
            if args.co_action:
                user_loc = coaction(loc_input, user_mlp)
                cat_loc = coaction(loc_input, cat_mlp)
                # cat_time = coaction(time_emb, cat_mlp)
                # user_cat = coaction(cat_input, user_mlp)
            else:
                user_loc = torch.cat((user_emb, loc_s_emb), 1)
            input_emb = torch.cat((user_emb, loc_s_emb, cat_s_emb, time_emb, user_loc, cat_time, cat_loc), 1)#
        else:
            input_emb = user_emb + cat_s_emb + loc_s_emb + timeslot_emb + dw_emb
        batch_input_emb.append(input_emb)
        batch_loc_labels.append(torch.tensor(loc_labels).to(device))
        batch_cat_labels.append(torch.tensor(cat_labels).to(device))
        src_lengths.append(len(data_b[i][0]))    
        labels.append(loc_labels[-1]) #对于验证集和测试集而言，是要预测的标签
        # prior_labels.append(loc[-1])   
        prior_loc.append(loc[-1]) 
        lengh.append(len(loc))       
    #shape:max_len * batch * hidden_dim        
    batch_pad_emb = pad_sequence(batch_input_emb, batch_first=False, padding_value=0)
    batch_loc_labels_pad = pad_sequence(batch_loc_labels, batch_first=False, padding_value=-1)
    batch_cat_labels_pad = pad_sequence(batch_cat_labels, batch_first=False, padding_value=-1)
    
    #增加_key_padding_mask
    key_pad_mask = torch.where(batch_loc_labels_pad==-1, True, False)

    return batch_pad_emb, batch_loc_labels_pad, batch_cat_labels_pad, labels, key_pad_mask, prior_loc, lengh, day_mode, week_mode
    # return batch_pad_emb, batch_loc_labels_pad, batch_cat_labels_pad, labels, key_pad_mask


def cal_acc_mrr(idxx, label, indices):
    acc = np.zeros((4, 1)) #top20 10 5 1
    mrr = 0
    for j, p in enumerate(idxx):
        t = label[j]
        if t in p:
            acc[0] += 1 #@20
            pos = np.argwhere(p == t)
            if pos >= 10:
                continue
            elif pos >= 5 and pos < 10:
                acc[1] += 1 #@10
            elif pos >= 1 and pos < 5:
                acc[1] += 1 #@10
                acc[2] += 1 #@5
            else:
                acc[1] += 1 #@10
                acc[2] += 1 #@5
                acc[3] += 1 #@1

    for i, loc_id in enumerate(label):
        id = np.argwhere(indices[i] == loc_id)
        mrr += 1 / (id + 1)
    acc_20 = acc[0] / len(label)
    acc_10 = acc[1] / len(label)
    acc_5 = acc[2] / len(label)
    acc_1 = acc[3] / len(label)
    mrr = mrr / len(label)
    return acc_1, acc_5, acc_10, acc_20, mrr

def evaluate(args, valid_loader, data, max_len, time_emb_model, user_emb_model, cat_emb_model, 
             loc_emb_model, geogcn_model, transformer_encoder_model, user_mlp_model, cat_mlp_model, loc_input_model, coaction, loss_fn, device):
    user_emb_model.eval()
    cat_emb_model.eval()
    loc_emb_model.eval()   
    user_mlp_model.eval()   
    cat_mlp_model.eval()   
    loc_input_model.eval()      
    geogcn_model.eval()
    transformer_encoder_model.eval()
    cat_emb = cat_emb_model(torch.tensor(range(data.cat_num)).to(device))
    loc_emb = loc_emb_model(torch.tensor(range(data.loc_num)).to(device))
    loc_emb = geogcn_model(loc_emb)
    loss_loc_list = []
    loss_cat_list = []
    loss_total_list = []
    acc_list_1 = [] #Top 1
    acc_list_5 = [] #Top 5
    acc_list_10 = []
    acc_list_20 = []
    mrr_list = []
    for b, data_b in enumerate(valid_loader): 
        batch_emb, loc_labels_emb, cat_labels_emb, label_loc, key_pad_mask, prior_loc, lengh, day_mode, week_mode = batch_seq_emb(args, data_b, max_len, cat_emb, loc_emb, time_emb_model, 
                                                                             user_emb_model, user_mlp_model, cat_mlp_model, loc_input_model, coaction, device)
        seq_out_loc, seq_out_cat = transformer_encoder_model(batch_emb, key_pad_mask, day_mode, week_mode)
        #计算loss 
        loss_loc = loss_fn(seq_out_loc.transpose(1, 2), loc_labels_emb)
        loss_cat = loss_fn(seq_out_cat.transpose(1, 2), cat_labels_emb)
        loss = loss_loc + args.cat_loss_w * loss_cat
        #计算ACC、MRR
        pred_loc = []
        for i in range(len(lengh)):
            tmp = seq_out_loc[lengh[i]-1, i, :]
            pred_loc.append(tmp)
        pred_loc = torch.stack(pred_loc)
        
        #添加先验概率修正
        if args.piror:
            trans_matrix = torch.tensor(data.trans_matrix.toarray()).to(device)
            correct = trans_matrix[prior_loc]
            pred_loc = pred_loc + correct 
               
        _, idxx = pred_loc.data.topk(20, dim=-1)
        idxx = idxx.detach().cpu().numpy()
        indices = torch.argsort(pred_loc, descending=True)
        indices = indices.detach().cpu().numpy()
        acc1, acc5, acc10, acc20, mrr = cal_acc_mrr(idxx, label_loc, indices)
        acc_list_1.append(acc1)
        acc_list_5.append(acc5)
        acc_list_10.append(acc10)
        acc_list_20.append(acc20)
        mrr_list.append(mrr)
        loss_loc_list.append(loss_loc.detach().cpu().numpy())
        loss_cat_list.append(loss_cat.detach().cpu().numpy())
        loss_total_list.append(loss.detach().cpu().numpy())
    epoch_acc1 = np.mean(acc_list_1)
    epoch_acc5 = np.mean(acc_list_5)
    epoch_acc10 = np.mean(acc_list_10)
    epoch_acc20 = np.mean(acc_list_20)
    epoch_mrr = np.mean(mrr_list)
    epoch_loss_loc = np.mean(loss_loc_list)
    epoch_loss_cat = np.mean(loss_cat_list)
    epoch_loss_total = np.mean(loss_total_list)
    return epoch_acc1, epoch_acc5, epoch_acc10, epoch_acc20, epoch_mrr, epoch_loss_loc, epoch_loss_cat, epoch_loss_total

'''
def cluster_center(loc_cat, loc_emb, device):
    cluster_emb = []
    for i in range(len(loc_cat)):
        cluster_emb.append(loc_emb[loc_cat[i]].mean(0))
    return torch.stack(cluster_emb)
'''

def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
    
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)
    # result_dir = increment_path('run-NYC/' + args.run_directory + str(args.enc_layer_num) +'-'+str(args.GeoGCN_layer_num)+'-'+ str(args.lr_patience), sep='-')
    result_dir = increment_path('run-NYC/' + args.run_directory + 'compare_p_dt', sep='-')
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    logging.Formatter.converter = beijing
    log_name=(datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d')
    logname = args.log_name + log_name + '.log'
    logging.basicConfig(
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=20,
        filename=os.path.join(result_dir, logname),
        filemode='a'
        ) 
    best_result = os.path.join(result_dir, 'best.txt')  
    f = open(best_result, 'a')
    f.write('-----------dataset:' + args.dataset +'----------------\n')
    f.write('enc_layer:'+' '+str(args.enc_layer_num)+' '+'GeoGCN_layer:'+' '+str(args.GeoGCN_layer_num)+' '+'lr:'+' '+str(args.lr)+' '+'weight_decay:'+' '+str(args.weight_decay)+' '+'enc_ffn_hdim:'+str(args.enc_ffn_hdim)+'\n')
    f.write('gcn_drop:'+' '+str(args.gcn_drop)+' '+'enc_drop:'+' '+str(args.enc_drop)+' '+'enc_nhead:'+' '+str(args.enc_nhead)+'\n')
    f.write('piror:'+' '+str(args.piror)+' '+'dec_time:'+' '+str(args.dec_time)+'\n')
    # f.write('early_patience:'+' '+str(args.patience)+' '+'lr_patience:'+' '+str(args.lr_patience)+' '+'is_lightgcn:'+' '+str(args.is_lightgcn)+' '+'is_sgc:'+' '+str(args.is_sgc)+'\n')
    f.write('early_patience:'+' '+str(args.patience)+' '+'lr_patience:'+' '+str(args.lr_patience)+' '+'hidden_dim:'+' '+str(args.hidden_dim)+' '+'time_dim:'+' '+str(args.time_dim)+'\n')
        
    logging.log(23, args)
    logging.log(23,f"---------------------dataset: {args.dataset}---------------------------------------")
    logging.log(23,f"enc_layer: {args.enc_layer_num} GeoGCN_layer: {args.GeoGCN_layer_num} lr: {args.lr} weight_decay: {args.weight_decay} enc_ffn_hdim: {args.enc_ffn_hdim}")
    logging.log(23,f"gcn_drop: {args.gcn_drop} enc_drop: {args.enc_drop} enc_nhead:{args.enc_nhead}")
    logging.log(23,f"piror: {args.piror} dec_time: {args.dec_time}")
    # logging.log(23,f"early_patience:{args.patience} lr_patience:{args.lr_patience} is_lightgcn:{args.is_lightgcn} is_sgc:{args.is_sgc}")
    logging.log(23,f"early_patience:{args.patience} lr_patience:{args.lr_patience} hidden_dim: {args.hidden_dim} time_dim: {args.time_dim}")
    if args.early_stop:
        stopper = EarlyStopping(args.patience)
        
    data = Data(args)
    co_action_w = args.co_action_w
    WEIGHT_EMB_DIM = sum([w[0]*w[1] for w in co_action_w])
    time_emb_model = SlotEncoding(args.time_dim, device=device)
    user_emb_model = nn.Embedding(data.user_num, args.hidden_dim).to(device)
    cat_emb_model = nn.Embedding(data.cat_num, args.time_dim).to(device) 
    loc_emb_model = nn.Embedding(data.loc_num, args.hidden_dim).to(device)
    if args.co_action:
        user_mlp_model = nn.Embedding(data.user_num, WEIGHT_EMB_DIM).to(device)
        cat_mlp_model = nn.Embedding(data.cat_num, WEIGHT_EMB_DIM).to(device)
        loc_input_model = nn.Embedding(data.loc_num, co_action_w[0][0]).to(device)
        coaction = Gen_Coaction(args)

    geogcn_model = GeoGCN(data.loc_g, data.tran_edge_weight, args, device).to(device)
    transformer_encoder_model = SeqPred(data.cat_num, data.loc_num, args, time_emb_model, device).to(device)
    optimizer = torch.optim.Adam(params=list(user_emb_model.parameters()) +
                                  list(cat_emb_model.parameters()) +
                                  list(loc_emb_model.parameters()) +
                                  list(user_mlp_model.parameters()) +
                                  list(cat_mlp_model.parameters()) +
                                  list(loc_input_model.parameters()) +
                                #   list(cat_input_model.parameters()) +
                                  list(geogcn_model.parameters()) +
                                  list(transformer_encoder_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', verbose=True, factor=args.lr_factor, patience=args.lr_patience, min_lr=1e-7)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    penalty_fn = nn.Softmax(dim=1)
    
    train_dataset = DatasetPrePare(data.train_forward, data.train_labels, data.train_user)
    train_loader = DataLoader(train_dataset, batch_size=args.batch,
                            shuffle=True, pin_memory=True, num_workers=0, collate_fn=lambda x:x)
    valid_dataset = DatasetPrePare(data.valid_forward, data.valid_labels, data.valid_user)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch,
                            shuffle=False, pin_memory=True, num_workers=0, collate_fn=lambda x:x)
    train_loss_loc_list = []
    train_loss_cat_list = []
    train_loss_total_list = []
    train_acc_list_1 = [] #Top 1
    train_acc_list_5 = [] #Top 5
    train_acc_list_10 = []
    train_acc_list_20 = []
    train_mrr_list = []
    
    start_time = time()
    last_time = start_time
    for epoch in range(args.epochs):
        user_emb_model.train()
        cat_emb_model.train()
        loc_emb_model.train()
        user_mlp_model.train()
        cat_mlp_model.train()
        loc_input_model.train()
        # cat_input_model.train()
        # interaction_model.train()
        geogcn_model.train()
        transformer_encoder_model.train()
        for b, data_b in enumerate(train_loader):
            cat_emb = cat_emb_model(torch.tensor(range(data.cat_num)).to(device))
            loc_emb = loc_emb_model(torch.tensor(range(data.loc_num)).to(device))
            loc_emb = geogcn_model(loc_emb)
            batch_emb, loc_labels_emb, cat_labels_emb, _, key_pad_mask,  _, _, day_mode, week_mode = batch_seq_emb(args, data_b, args.traj_max_len, cat_emb, loc_emb, time_emb_model, 
                                                                                 user_emb_model, user_mlp_model, cat_mlp_model, loc_input_model, coaction, device)

            seq_out_loc, seq_out_cat = transformer_encoder_model(batch_emb, key_pad_mask, day_mode, week_mode)
            
            loss_loc = loss_fn(seq_out_loc.transpose(1, 2), loc_labels_emb)#具体位置交叉熵
            loss_cat = loss_fn(seq_out_cat.transpose(1, 2), cat_labels_emb)#语义类别交叉熵
            # '''
            test_time = time()
            tmp = torch.where(loc_labels_emb>=0, 1, loc_labels_emb)#pad部分:-1，标签部分：1
            loc_sm = penalty_fn(seq_out_loc.transpose(1, 2))
            loc_pos = torch.argmax(loc_sm, dim=1)
            loc_pos = torch.where(loc_pos==0, 5081, loc_pos)#把0替换为位置数量
            loc_pos = loc_pos * tmp#将预测的位置index提取出来[40,1500]
            loc_pos = torch.where(loc_pos==5081, 0, loc_pos)
            #将预测的位置矩阵映射为类别矩阵矩阵
            loc_cat = torch.tensor(data.loc_cat).to(device)
            loc_map_cat = torch.where(loc_pos<0, loc_pos, loc_cat[loc_pos])
            loc_map_cat = torch.where(loc_map_cat<0, -1, loc_map_cat)
            cat_sm = penalty_fn(seq_out_cat.transpose(1, 2))
            cat_pos = torch.argmax(cat_sm, dim=1)
            cat_pos = torch.where(cat_pos==0, 32, cat_pos)
            cat_pos = cat_pos * tmp#将预测的类别index提取出来[40,1500]
            cat_pos = torch.where(cat_pos==32, 0, cat_pos)
            cat_pos = torch.where(cat_pos<0, -1, cat_pos)
            n_tmp = torch.where(tmp<0, 0, tmp)
            contradiction = torch.eq(loc_map_cat, cat_pos)
            contradiction = contradiction * n_tmp
            # loc_p_cat_right = torch.eq(loc_map_cat, cat_labels_emb)
            # loc_p_cat_right = loc_p_cat_right * n_tmp
            # penalty_w = ~torch.eq(contradiction, loc_p_cat_right)#False:直接预测的位置对应的类别正确，直接预测的类别与上述类别不一致，对其相应的cat_loss惩罚
            # p_cat_labels = penalty_w * cat_labels_emb
            # p_cat_labels = torch.where(p_cat_labels==0, -1, p_cat_labels)
            # penalty_loss_cat = loss_fn(seq_out_cat.transpose(1, 2), p_cat_labels)
            
            # cat_right = torch.eq(cat_pos, cat_labels_emb)
            # cat_right = cat_right * n_tmp
            # penalty_l = ~torch.eq(contradiction, cat_right)#False:直接预测的类别正确，直接预测的位置对应的类别与上述类别不一致，对其相应的loc_loss惩罚
            # p_loc_labels = penalty_l * loc_labels_emb
            # p_loc_labels = torch.where(p_loc_labels==0, -1, p_loc_labels)
            # penalty_loss_loc = loss_fn(seq_out_loc.transpose(1, 2), p_loc_labels)
            
            pp_cat_labels = contradiction * cat_labels_emb
            pp_loc_labels = contradiction * loc_labels_emb
            pp_cat_labels = torch.where(pp_cat_labels==0, -1, pp_cat_labels)
            pp_loc_labels = torch.where(pp_loc_labels==0, -1, pp_loc_labels)
            pen_cat = loss_fn(seq_out_cat.transpose(1, 2), pp_cat_labels)
            pen_loc = loss_fn(seq_out_loc.transpose(1, 2), pp_loc_labels)
            # '''  
            loss = loss_loc + loss_cat + pen_cat + pen_loc
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            """
            #计算ACC、MRR
            pred_loc = []
            for i in range(len(lengh)):
                tmp = seq_out_loc[lengh[i]-1, i, :]
                pred_loc.append(tmp)
            pred_loc = torch.stack(pred_loc)  
            _, idxx = pred_loc.data.topk(20, dim=-1)
            idxx = idxx.detach().cpu().numpy()
            indices = torch.argsort(pred_loc, descending=True)
            indices = indices.detach().cpu().numpy()
            acc1, acc5, acc10, acc20, mrr = cal_acc_mrr(idxx, label_loc, indices)
            train_acc_list_1.append(acc1)
            train_acc_list_5.append(acc5)
            train_acc_list_10.append(acc10)
            train_acc_list_20.append(acc20)
            train_mrr_list.append(mrr)
            train_loss_loc_list.append(loss_loc.detach().cpu().numpy())
            train_loss_cat_list.append(loss_cat.detach().cpu().numpy())
            train_loss_total_list.append(loss.detach().cpu().numpy())
            """
        # epoch_train_acc1 = np.mean(train_acc_list_1)
        # epoch_train_acc5 = np.mean(train_acc_list_5)
        # epoch_train_acc10 = np.mean(train_acc_list_10)
        # epoch_train_acc20 = np.mean(train_acc_list_20)
        # epoch_train_mrr = np.mean(train_mrr_list)
        # epoch_train_loss_loc = np.mean(train_loss_loc_list)
        # epoch_train_loss_cat = np.mean(train_loss_cat_list)
        # epoch_train_loss = np.mean(train_loss_total_list)
        # torch.cuda.empty_cache()
        val_perf = evaluate(args, valid_loader, data, args.traj_max_len, time_emb_model, user_emb_model, cat_emb_model, 
                            loc_emb_model, geogcn_model, transformer_encoder_model, user_mlp_model, cat_mlp_model, loc_input_model, coaction, loss_fn, device)
        monitor_loss = val_perf[-1]
        monitor_score = val_perf[0] + val_perf[3]
        score = np.sum(val_perf[0:4])
        lr_scheduler.step(monitor_score)
        # lr_scheduler.step(monitor_loss)

        if epoch % args.print_interval == 0:
            duration = time() - last_time
            last_time = time()
            # logging.log(23,f"Epoch {epoch}  train:  loss = {epoch_train_loss:.6f} Acc@20:{epoch_train_acc20:.4f} Acc@10:{epoch_train_acc10:.4f} Acc@5:{epoch_train_acc5:.4f} Acc@1:{epoch_train_acc1:.4f} MRR:{epoch_train_mrr:.4f} {duration:.3f} sec")        
            logging.log(23,f"Epoch {epoch} valid:loss_total = {val_perf[-1]:.6f} valid:loss_cat = {val_perf[-2]:.6f} valid:loss_loc = {val_perf[-3]:.6f} Score: {np.sum(val_perf[0:4]):.4f} Acc@20:{val_perf[3]:.4f} Acc@10:{val_perf[2]:.4f} Acc@5:{val_perf[1]:.4f} Acc@1:{val_perf[0]:.4f} MRR:{val_perf[4]:.4f} {duration:.3f} sec")    
        
        if args.early_stop and epoch != 0:
            if stopper.step(score, monitor_loss, user_emb_model, cat_emb_model, loc_emb_model, geogcn_model, transformer_encoder_model, user_mlp_model, cat_mlp_model, loc_input_model, epoch, result_dir):
                break  

    runtime = time() - start_time
    torch.cuda.empty_cache()
    if args.early_stop:
        logging.log(21, f"best epoch: {stopper.best_epoch}, best val acc:{stopper.best_score * 100:.4f}, val_loss:{stopper.best_epoch_val_loss:.6f}, ({runtime:.3f} sec)")
    if args.early_stop:
        state_dict = torch.load(os.path.join(result_dir, 'checkpoint.pt'))
        user_emb_model.load_state_dict(state_dict['user_emb_model_state_dict'])   
        cat_emb_model.load_state_dict(state_dict['cat_emb_model_state_dict'])   
        loc_emb_model.load_state_dict(state_dict['loc_emb_model_state_dict'])   
        user_mlp_model.load_state_dict(state_dict['user_mlp_model_state_dict'])   
        cat_mlp_model.load_state_dict(state_dict['cat_mlp_model_state_dict'])   
        loc_input_model.load_state_dict(state_dict['loc_input_model_state_dict'])      
        geogcn_model.load_state_dict(state_dict['geogcn_model_state_dict'])   
        transformer_encoder_model.load_state_dict(state_dict['transformer_encoder_model_state_dict'])    
    test_dataset = DatasetPrePare(data.test_forward, data.test_labels, data.test_user)
    test_loader = DataLoader(test_dataset, batch_size=args.batch,
                            shuffle=False, pin_memory=True, num_workers=0, collate_fn=lambda x:x)
    test_perf = evaluate(args, test_loader, data, args.traj_max_len, time_emb_model, user_emb_model, cat_emb_model, 
                         loc_emb_model, geogcn_model, transformer_encoder_model, user_mlp_model, cat_mlp_model, loc_input_model, coaction, loss_fn, device)
    logging.log(23,f"test: loss_total = {val_perf[-1]:.6f} loss_cat = {val_perf[-2]:.6f} loss_loc = {val_perf[-3]:.6f} Acc@20:{test_perf[3]:.4f} Acc@10:{test_perf[2]:.4f} Acc@5:{test_perf[1]:.4f} Acc@1:{test_perf[0]:.4f} MRR:{test_perf[4]:.4f} {runtime:.3f} sec")    
    f.write('valid: Best epoch: '+str(stopper.best_epoch) +' score: ' +str(float('%0.4f'%(stopper.best_score*100)))+' loss: '+str(stopper.best_epoch_val_loss)+'\n')
    f.write('test Acc@20: '+str(float('%0.4f'%test_perf[3]))+' Acc@10: '+str(float('%0.4f'%test_perf[2]))+' Acc@5: '+str(float('%0.4f'%test_perf[1]))+' Acc@1: '+str(float('%0.4f'%test_perf[0]))+' MRR: '+str(float('%0.4f'%test_perf[4]))+'\n')
    f.write(str(float('%0.2f'%runtime))+'sec'+'\n')
    f.close()       
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic')
    parser.add_argument('--dataset', type=str, default='./data/FS_NYC', help='FS_NYC, FS_TKY')
    parser.add_argument("--batch", type=int, default=1500, help="TransformerEncoder batch")
    parser.add_argument("--traj_max_len", type=int, default=40, help="traj max length")
    parser.add_argument("--dist", type=int, default=1000, help="500, 1000, 1500, 2000")
    parser.add_argument("--enc_nhead", type=int, default=2, help="TransformerEncoderLayer Attention head number")
    parser.add_argument("--enc_ffn_hdim", type=int, default=1024, help="TransformerEncoderLayer FFN hidden dim")
    parser.add_argument("--gcn_drop", type=float, default=0, help="GeoGCN dropout probability")
    parser.add_argument("--enc_drop", type=float, default=0.2, help="Encoder dropout probability")
    parser.add_argument('--hidden_dim', type=int, default=128, help='Model Layer connection dim')          
    parser.add_argument('--time_dim', type=int, default=32, help='Model Layer connection dim')          
    parser.add_argument('--enc_layer_num', type=int, default=1, help='Number of TransformerEncoder layers.')     
    parser.add_argument('--GeoGCN_layer_num', type=int, default=2, help='Number of Conv layers.')     
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--lr_factor', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=20, help='Patience in early stopping')
    parser.add_argument('--lr_patience', type=int, default=5, help='Patience in early stopping')
    parser.add_argument('--cat_loss_w', type=int, default=1, help='cat-loss weight')
    parser.add_argument('--interaction', type=int, default=1, help='interaction or not')
    parser.add_argument('--seed', type=int, default=42, help="seed for our system")
    parser.add_argument('--print_interval', type=int, default=1, help="the interval of printing in training")
    parser.add_argument('--week', type=int, default=1, help="the interval of printing in training")
    parser.add_argument('--day', type=int, default=1, help="the interval of printing in training")
    parser.add_argument('--time_wd', type=int, default=1, help="the interval of printing in training")
    parser.add_argument('--log_name', type=str, default='log', help="Name for logging")
    parser.add_argument('--run_directory', type=str, default='Debug', help="run directory")
    parser.add_argument('--early_stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--is_lightgcn", action='store_true', default=True, help="whether to use LightGCN")
    parser.add_argument("--is_att", action='store_true', default=True, help="whether to use attention to fuse loc features")
    parser.add_argument("--is_sgc", action='store_true', default=True, help="whether to use simple GCN")
    parser.add_argument("--co_action", action='store_true', default=True, help="whether to use co-action interaction")
    parser.add_argument("--co_action_w", type=list, default=[[16, 8], [8, 4]], help="ca-action weight")
    parser.add_argument('--piror', type=int, default=1, help="whether to use piror prob to correct the prediction")
    parser.add_argument('--dec_time', type=int, default=1, help="whether to use time in the category decoder of transformer")
    args = parser.parse_args()
       
    main(args)