import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import pickle as pkl
import torch.nn as nn
import torch
import pandas as pd
from LoadData3 import loadData
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from transformers import BertModel
from visModel import vgg
from evaluate import evaluation2class
from torch.autograd import Variable
from earlystopping import EarlyStopping
import time
import psutil
# from  info_nce import InfoNCE
# from torch_geometric.nn import GCNConv

import random
import pynvml
import sys
import time
'''
def wait_gpu(need_MEM_G = 20):
    pynvml.nvmlInit()
    while 1:
        for i in range(8):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 指定GPU的id
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print(meminfo.used / 1024 ** 3)
            # print(meminfo.free / 1024 ** 3)
            mem = meminfo.free / 1024 ** 3
            if mem > need_MEM_G:
                #os.environ['CUDA_VISIBLE_DEVICES'] = i
                return str(i)
        #print('No GPU is empty')
        time.sleep(2)

os.environ['CUDA_VISIBLE_DEVICES'] = wait_gpu()
'''

# pid_list = psutil.pids()
# wait_pid = 2720  # 等待的进程号
# while wait_pid in pid_list:
#     #print('still working!' + str(wait_pid))
#     time.sleep(2)
#     pid_list = psutil.pids()

def get_non_pad_mask(seq):
    # assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)

# =========================================================================================
def get_attn_key_pad_mask(seq_k, seq_q):
    # For masking out the padding part of key sequence.
    # Expand to fit the shape of key query attention matrix.
    # print('seq_k: ',seq_k.shape) [8, 305, 256]
    len_q = seq_q.size(1)
    # print('seq_k,ne(0): ', seq_k.ne(0)) [8, 305, 256]
    padding_mask = seq_k.ne(0)  # Constants.PAD seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # print("q:",q.shape) q: torch.Size([8, 4, 305, 64])
        # print("k:", k.shape) k: torch.Size([8, 4, 305, 64])
        # print("q / self.temperature:", (q / self.temperature).shape) q / self.temperature: torch.Size([8, 4, 305, 64])
        # print("k.transpose(2, 3):", (k.transpose(2, 3)).shape) k.transpose(2, 3): torch.Size([8, 4, 64, 305])

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # print("attn.shape: ",attn.shape)  #[8,4,305,305]
        # mask [8,1,305,256]
        # attn = torch.matmul(q, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHead(nn.Module):

    def __init__(self, n_head, d_model, d_qkv, dropout=0.3):
        super().__init__()

        self.n_head = n_head
        self.d_qkv = d_qkv

        self.w_qs = nn.Linear(d_model, n_head * d_qkv, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_qkv, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_qkv, bias=False)

        self.fc = nn.Linear(n_head * d_qkv, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_qkv ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_qkv, n_head = self.d_qkv, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_qkv)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_qkv)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_qkv)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None: mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        # q = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q  # , attn


class MultiHeadLayer(nn.Module):
    def __init__(self, n_head, d_model, d_qkv, dropout=0.3):
        super(MultiHeadLayer, self).__init__()
        self.multi_head = MultiHead(n_head, d_model, d_qkv, dropout=dropout)

    def forward(self, seq_q, seq_k, q, k, v):
        if seq_q is None and seq_k is None:
            enc = self.multi_head(q=q, k=k, v=v)
        else:
            att_mask = get_attn_key_pad_mask(seq_k=seq_k, seq_q=seq_q)
            pad_mask = get_non_pad_mask(seq=seq_q)

            enc = self.multi_head(q=q, k=k, v=v, mask=att_mask)
            enc *= pad_mask

            del att_mask, pad_mask

        return enc


class SelfAttentionSubEncoder(nn.Module):
    def __init__(self, n_head, d_model, d_qkv, dropout):
        super(SelfAttentionSubEncoder, self).__init__()

        self.mha_layer = MultiHeadLayer(n_head, d_model, d_qkv, dropout)
        self.ffn_layer = hard_fc(d_model, d_model, dropout)

    def forward(self, x, x_enc):
        enc = self.mha_layer(x, x, x_enc, x_enc, x_enc)
        enc = self.ffn_layer(enc)

        if x is not None:
            enc *= get_non_pad_mask(seq=x)

        return enc


class FocalLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', gamma=2, eps=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp * self.eps
        return loss.mean()


# Transformer FFN
class hard_fc(torch.nn.Module):
    def __init__(self, d_in, d_hid, DropOut=0.5):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(DropOut)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name='hard_fc'):  # T15: epsilon = 0.2
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='hard_fc'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup  # 这句的意思name在字典中
                param.data = self.backup[name]
        self.backup = {}

class PGD():
    def __init__(self, model, eps=0.1, alpha=0.3):
        #super(PGD, self).__init__(model, eps)
        self.alpha = alpha
        self.eps = eps
        self.model = model
        self.grad_backup = {}
        self.emb_backup = {}

    def attack(self, emb_name='hard_fc', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    # FGD对抗扰动
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    # 约束
                    param.data = self.project(name, param.data)

    def project(self, emb_name, param_data):
        r = param_data - self.emb_backup[emb_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[emb_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class MF_Net(torch.nn.Module):
    def __init__(self, opt):
        super(MF_Net, self).__init__()
        self.bert = BertModel.from_pretrained(opt.bert_path)
        # self.vgg = torch.nn.DataParallel(vgg(model_name="vgg19"))
        self.vgg = vgg(model_name="vgg19")
        # self.fc = torch.nn.Linear(2*opt.out_feats, 2)
        # self.pca = PCA(n_components=256)
        # self.embedding_dim = nn.Linear(768, opt.d_model)
        # 增加一个多头注意力模块
        # self.self_attn_layers_list = nn.ModuleList(
        #         [SelfAttentionSubEncoder(opt.n_head, opt.d_model, opt.d_qkv, opt.dropout_mha) for _ in range(opt.num_layers_mha)])
        self.lstm1 = nn.LSTM(768, 768, batch_first=True, bidirectional=True) # 原384
        self.lstm2 = nn.LSTM(1, 768, batch_first=True, bidirectional=True) # 原384

        self.hard_fc1 = hard_fc(opt.out_feats, opt.out_feats)
        self.hard_fc2 = hard_fc(opt.out_feats, opt.out_feats)
        self.fc = torch.nn.Sequential(  # 三层全连接层
            nn.Dropout(p=0.5),  # 减少过拟合 以50%的概率随机失活神经元
            nn.Linear(768 * 4, 768),  # 第二层全联接层  他的输入是上一层的输出节点个数
            nn.ReLU(True),
            nn.Dropout(p=0.5),  # 随机失活
            nn.Linear(768, 2)  # classnum是最终需要分类的类别个数
        )
        self.w1 = nn.Linear(768, 1)
        self.w2 = nn.Linear(768, 1)
        self.w1_2 = nn.Linear(768, 1)
        self.w2_2 = nn.Linear(768, 1)
        # self.cl_loss = InfoNCE()
        # self.ffn = hard_fc(opt.out_feats, opt.out_feats)
        # self.hard_fc2 = hard_fc(opt.out_feats, opt.out_feats)  # optional
        # self.text_feature = None

    def init_hidden(self, batch_size, hidden_dim):
        h = Variable(torch.zeros((2, batch_size, hidden_dim)))
        c = Variable(torch.zeros((2, batch_size, hidden_dim)))
        h, c = h.to(device), c.to(device)
        return h, c

    def get_clloss(self, T, x, y):
        n = y.shape[0]
        # 下面这部分是对比学习的
        T = T  # pheme: t = 0.6
        # 这步得到它的相似度矩阵
        if torch.sum(torch.sum(x != x, dim=1), dim=0).detach().cpu() != 0:
            print('x出现了nan值')
            exit(-1)

        similarity_matrix = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
        if torch.sum(torch.sum(similarity_matrix != similarity_matrix, dim=1), dim=0).detach().cpu() != 0:
            print('similarity_matrix 出现了nan值')
            exit(-1)

        # 这步得到它的label矩阵，相同label的位置为1
        mask = torch.ones_like(similarity_matrix) * (y.expand(n, n).eq(y.expand(n, n).t()))
        mask.to(device)

        # 这步得到它的不同类的矩阵，不同类的位置为1
        mask_no_sim = torch.ones_like(mask) - mask
        mask_no_sim.to(device)
        # 这步产生一个对角线全为0的，其他位置为1的矩阵
        mask_dui_jiao_0 = torch.ones(n, n) - torch.eye(n, n)
        mask_dui_jiao_0 = mask_dui_jiao_0.to(device)

        # 这步给相似度矩阵求exp,并且除以温度参数T

        similarity_matrix = torch.exp(similarity_matrix / T).to(device)
        # print(similarity_matrix)

        # 这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
        # print("similarity_matrix***: ",similarity_matrix)
        similarity_matrix = similarity_matrix * mask_dui_jiao_0
        similarity_matrix.to(device)
        # print("对角化后similarity_matrix***: ",similarity_matrix)

        # 这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        sim = mask * similarity_matrix

        # 用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
        no_sim = similarity_matrix - sim

        # 把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim, dim=1)

        # 将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        # 至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
        # 每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。

        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum = sim + no_sim_sum_expend
        # print("sim: ", sim)
        # print("sim_sum: ", sim_sum)
        if torch.sum(torch.sum(sim_sum != sim_sum, dim=1), dim=0).detach().cpu() != 0:
            print('分母出现了nan值')
            exit(-1)
        loss = torch.div(sim, sim_sum).to(device)
        if torch.sum(torch.sum(sim_sum != sim_sum, dim=1), dim=0).detach().cpu() != 0:
            print('...chufa chuxian1 le1 nan值')
            exit(-1)
        # print("loss: ", loss)

        # 由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        # 全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。

        loss = mask_no_sim + loss + torch.eye(n, n).to(device)
        loss = loss.to(device)

        # 接下来就是算一个批次中的loss了
        loss = -torch.log(loss)  # 求-log
        # print("loss: ", loss)
        cl_loss = torch.sum(torch.sum(loss, dim=1)) / (2 * n)  # 将所有数据都加起来除以2n
        if torch.isnan(cl_loss.detach().cpu()):
            print('cl_loss出现了nan值')
            cl_loss = torch.tensor(0)
        return cl_loss

    def forward(self, input_ids1, attention_mask1, token_type_ids1, img1, labels1, comm, sen):
        # print("intput_ids: ",input_ids1.shape)
        outputs1 = self.bert(
            input_ids1,
            attention_mask=attention_mask1,
            token_type_ids=token_type_ids1,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None)[0][:, 0, :]
        
        img_guide_out1, _ = self.vgg(img1)

        x1 = torch.cat((outputs1, img_guide_out1), 1)  # 两种表征拼接
        x1 = self.hard_fc1(x1)

        x = x1  # 把原数据和增强数据在行上拼接
        y = labels1  # y1是原数据的标签，y2是增强数据的标签
        cl_loss1 = self.get_clloss(opt.T1, x, y)

        h0, c0 = self.init_hidden(comm.size(0), 768) # 原为384
        output_1, (h, c) = self.lstm1(comm, (h0, c0))
        # print("sen.shape: ", sen.shape)
        sen = sen.unsqueeze(2)
        h0_2, c0_2 = self.init_hidden(sen.size(0), 768) # 原为384
        output_2, (h_2, c_2) = self.lstm2(sen, (h0_2, c0_2))
        #new_h,new_h_1 = h[0,:,:],h[-1,:,:]
        output_1 = output_1[:,-1,:].squeeze(1)

        output_2 = output_2[:,-1,:].squeeze(1)

        '''可注释'''
        F1 = output_1 * output_2
        # print("F1.shape: ",F1.shape)
        f1 = F.softmax(torch.tanh(self.w1(output_1) + self.w2(F1)))
        # print("f1.shape: ",f1.shape)
        f1_2 = F.softmax(torch.tanh(self.w1(output_2) + self.w2(F1)))
        # print("f1_2.shape: ",f1_2.shape)
        Z = f1 * output_1 + f1_2 * output_2
        Z = torch.sum(Z, dim=1)

        '''
        Z = torch.cat((output_1, output_2), 1)  # 两种表征拼接
        Z = self.hard_fc2(Z)
        '''
        cl_loss2 = self.get_clloss(opt.T2,Z,y)
        x = torch.cat((x, Z), 1)
        # print("x: ",x.shape)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        cl_loss = cl_loss1 + 0.1 * cl_loss2
        print("cl_loss1: ",cl_loss1.item())
        print("cl_loss2: ",cl_loss2.item())
        return x, cl_loss, y


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


def train(opt, x_train,  x_test, lr, weight_decay, patience, n_epochs, batchsize, dataname):
    MF_model = MF_Net(opt).to(device)
    MF_model = torch.nn.DataParallel(MF_model, device_ids=[0]).module
    #focalloss = FocalLoss(gamma=opt.gamma,eps = opt.eps)

    #fgm = FGM(MF_model)
    fgm = PGD(MF_model)
    for para in MF_model.hard_fc1.parameters():
        para.requires_grad = False
    for para in MF_model.hard_fc2.parameters():
        para.requires_grad = False

    bert_params = []
    # attn_params = []
    # output_params = []
    other_params = []

    for name, parms in MF_model.named_parameters():
        if "bert" in name or 'vgg' in name:
             bert_params += [parms]
        else:
             if parms.requires_grad:
                 other_params += [parms]
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, MF_model.parameters()), lr=lr,
    #                             weight_decay=weight_decay)
    optimizer = torch.optim.Adam([
         {"params": bert_params, 'lr':opt.bert_optim_lr}, #bert层参数被冻结
         {"params": other_params, 'lr': opt.other_optim_lr}], weight_decay = 3e-4)
    #     {"params": other_params, 'lr': 5e-5},
    # ], weight_decay=3e-4,)

    # optional ------ S1 ----------
    for para in MF_model.hard_fc1.parameters():
        para.requires_grad = True

    for para in MF_model.hard_fc2.parameters():
        para.requires_grad = True

    # print("============输出模型的参数信息=========")
    # for name, value in MF_model.named_parameters():
    #    print('name: {}'.format(name))

    # text_model.train()
    # MF_model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_acc = 0.0
    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    traindata_list, valdata_list = loadData(dataname, x_train,x_test)  # T15 droprate = 0.1
    train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=4)
    val_loader = DataLoader(valdata_list, batch_size=batchsize, shuffle=False, num_workers=4)

    train_process_datas = []
    for epoch in range(n_epochs):
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        NUM = 1
        beta = 0.001
        MF_model.train()
        for Batch_data in train_loader:
            # print(type(Batch_data))
            data1 = Batch_data
            input_ids1 = data1['input_ids'].to(device)
            comm = data1['comm'].to(device)
            sen = data1['sen'].to(device)
            attention_mask1 = data1['attention_mask'].to(device)
            labels1 = data1['labels'].to(device)
            token_type_ids1 = data1['token_type_ids'].to(device)
            img1 = data1['img'].to(device)
            #print('shape_input_ids:', input_ids1.shape)
            #print('img shape:',img1.shape)
            #print('label shape:',labels1.shape)
            #print("comm.shape:",comm.shape)
            #print("sen.shape: ",sen.shape)
            out_labels, cl_loss, y = MF_model(input_ids1, attention_mask1, token_type_ids1, img1, labels1, comm, sen)
            # out_labels, cl_loss, y = img_model(img1, img2, labels1, labels2)

            finalloss = F.nll_loss(out_labels, y)
            #finalloss = focalloss(out_labels, y)
            # print("finalloss: ",finalloss.item())
            line1 = "cl_loss: " + str(cl_loss.item()) + '\n'
            train_process_datas.append(line1)
            print("cl_loss: ", cl_loss.item())
            loss = finalloss + beta * cl_loss
            # print("loss.item: ",loss.item())
            avg_loss.append(loss.item())

            # S1和S2应该是指两种对抗训练的方法
            ##------------- S1 ---------------##
            # 对抗训练
            '''
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            avg_loss.append(loss.item())
            optimizer.step()
            epsilon = 3
            loss_ad = epsilon/(finalloss + 0.001*cl_loss)
            print('loss_ad: ', loss_ad)
            optimizer_hard.zero_grad()
            loss_ad.backward()
            optimizer_hard.step()
            '''
            ##--------------------------------##
            ##------------- S2 ---------------##
            optimizer.zero_grad()
            loss.backward()  # 反向传播，得到正常的grad
            fgm.backup_grad()
            for t in range(5):
                fgm.attack(is_first_attack=(t==0))  # 更新FAN 的参数  # 在embedding上添加对抗扰动
                if t != 4:
                    optimizer.zero_grad()
                else:
                    fgm.restore()
                out_labels, cl_loss, y = MF_model(input_ids1, attention_mask1, token_type_ids1, img1, labels1, comm, sen)
                finalloss = F.nll_loss(out_labels, y)
                #finalloss = focalloss(out_labels, y)
                loss_adv = finalloss + beta * cl_loss
                loss_adv.backward()
            fgm.restore()  # 更新复原 的参数
            optimizer.step()
            ##--------------------------------##
            ##------------- S2 ---------------##

            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(y).sum().item()
            train_acc = correct / len(y)
            avg_acc.append(train_acc)
            line2 = "Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc) + '\n'
            train_process_datas.append(line2)
            print("Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            sys.stdout.flush()
            batch_idx = batch_idx + 1
            NUM += 1
            # print('train_loss: ', loss.item())
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2 = [], [], [], [], [], [], [], [], []
        MF_model.eval()
        tqdm_val_loader = tqdm(val_loader)
        for Batch_data in tqdm_val_loader:
            data1 = Batch_data
            input_ids1 = data1['input_ids'].to(device)
            comm = data1['comm'].to(device)
            sen = data1['sen'].to(device)
            # print('shape_input_ids:', input_ids.shape)
            attention_mask1 = data1['attention_mask'].to(device)
            labels1 = data1['labels'].to(device)
            token_type_ids1 = data1['token_type_ids'].to(device)
            img1 = data1['img'].to(device)

            val_out, val_cl_loss, y = MF_model(input_ids1, attention_mask1, token_type_ids1, img1, labels1, comm, sen)

            val_loss = F.nll_loss(val_out, y)
            #val_loss = focalloss(val_out, y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(y).sum().item()
            val_acc = correct / len(y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluation2class(
                val_pred, y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        line3 = "Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                            np.mean(temp_val_accs)) + '\n'
        train_process_datas.append(line3)
        print("Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                            np.mean(temp_val_accs)))
        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2))]
        line4 = " ".join(res)
        line4 = 'Val results:' + line4 + '\n'
        train_process_datas.append(line4)
        print('Val results:', res)
        sys.stdout.flush()
        tmp_acc = np.mean(temp_val_Acc_all)
        if tmp_acc > best_acc:
            best_acc = tmp_acc
            #torch.save(MF_model.state_dict(), opt.model_file)
            torch.save(MF_model, opt.model_file)
            line5 = "模型已保存\n\n"
            train_process_datas.append(line5)
            print("模型已保存")
        else:
            line5 = "该模型不是最优模型\n\n"
            train_process_datas.append(line5)
            print("该模型不是最优模型")
        '''
        if epoch > 15:
            early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                           MF_model, 'My', dataname)
        '''
        accs = np.mean(temp_val_accs)
        pre1 = np.mean(temp_val_Prec1)
        rec1 = np.mean(temp_val_Recll1)
        F1 = np.mean(temp_val_F1)
        pre2 = np.mean(temp_val_Prec2)
        rec2 = np.mean(temp_val_Recll2)
        F2 = np.mean(temp_val_F2)
        '''
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            break
        '''
           
    line6 = "最优模型在测试集上的准确率为:" + str(best_acc) + '\n\n'
    train_process_datas.append(line6)
    
    with open('./train_process.txt','w') as f:
        f.writelines(train_process_datas)
    return accs, pre1, rec1, F1, pre2, rec2, F2

##---------------------------------main---------------------------------------
def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--bert_path', default='./autodl-fs/bert-base-chinese',
                        help='preTrained director.')
    parser.add_argument('--out_feats', default=768*2,
                        help='preTrained director.')
    parser.add_argument('--batchsize', default=8,
                        help='preTrained director.')
    parser.add_argument('--lr', default=2e-5,
                        help='preTrained director.')
    parser.add_argument('--n_epoch', default=15,
                        help='preTrained director.')
    parser.add_argument('--T1', default=0.7,
                        help='preTrained director.')
    parser.add_argument('--T2', default=0.4,
                        help='preTrained director.')
    parser.add_argument('--dataset', default='weibo',
                        help='preTrained director.')

    parser.add_argument('--model_file', default='./autodl-tmp/model/weibo',
                        help='preTrained director.')
    parser.add_argument('--gamma', default= 2,
                        help='preTrained director.')
    parser.add_argument('--eps', default=0.3,
                        help='preTrained director.')
    # optimization
    parser.add_argument('--bert_optim_lr', type=float, default=2e-5,
                        help='bert learning rate')
    parser.add_argument('--other_optim_lr', type=str, default=6e-4,
                        help='pca learning rate')
    # atten

    parser.add_argument('--d_model', type=int, default=768,
                        help='using cosine annealing')
    parser.add_argument('--n_head', type=int, default=8,
                        help='warm-up for large batch training')
    parser.add_argument('--d_qkv', type=int, default=96,
                        help='id for recording multiple runs')
    parser.add_argument('--dropout_mha', type=float, default=0.5,
                        help='id for recording multiple runs')
    parser.add_argument('--num_layers_mha', type=int, default=3,
                        help='id for recording multiple runs')

    opt = parser.parse_args()
    return opt

def set_seed(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



set_seed(2022)

opt = parse_option()
# opt.model_file = './model/{}/model.pth.tar'.format(datasetname)
print(str(opt))
scale = 1
lr = opt.lr
weight_decay = 1e-6
patience = 50
n_epochs = opt.n_epoch
batchsize = opt.batchsize
datasetname = opt.dataset  # (1)Twitter15  (2)pheme  (3)weibo

if datasetname == 'weibo':
    opt.bert_path = './autodl-fs/bert-base-chinese' 
if datasetname == "fakeddit":
    n_epochs = 3

opt.model_file = './autodl-tmp/model/{}/model.pth.tar'.format(datasetname)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_accs = []
NR_F1 = []  # NR
R_F1 = []  # R
NR_pre = []
R_pre = []
NR_rec = []
R_rec = []

dataPath = './autodl-fs/data/{}/rand5fold.pkl'.format(datasetname)

fold0_x_test, fold0_x_train, \
fold1_x_test, fold1_x_train, \
fold2_x_test, fold2_x_train, \
fold3_x_test, fold3_x_train, \
fold4_x_test, fold4_x_train = pkl.load(open(dataPath,"rb"))

# train(opt, train_data, val_data, test_data, lr, weight_decay, patience, n_epochs, batchsize, datasetname)
accs_0, pre1_0, rec1_0, F1_0, pre2_0,rec2_0,F2_0 = train(opt,fold0_x_train, fold0_x_test, lr, weight_decay, patience, n_epochs, batchsize, datasetname)
#lr = lr *0.1
#accs_1, pre1_1, rec1_1, F1_1, pre2_1,rec2_1,F2_1 = train(opt,fold1_x_train, fold1_x_test, lr, weight_decay, patience, n_epochs, batchsize,datasetname)
#lr = lr * 0.1
#accs_2, pre1_2, rec1_2, F1_2, pre2_2,rec2_2,F2_2  = train(opt,fold2_x_train, fold2_x_test, lr, weight_decay, patience, n_epochs, batchsize,datasetname)
#lr = lr*0.1
#accs_3, pre1_3, rec1_3, F1_3, pre2_3,rec2_3,F2_3 = train(opt,fold3_x_train, fold3_x_test, lr, weight_decay, patience, n_epochs, batchsize, datasetname)
#lr = lr*0.1
#accs_4, pre1_4, rec1_4, F1_4, pre2_4,rec2_4,F2_4 = train(opt,fold4_x_train, fold4_x_test, lr, weight_decay, patience, n_epochs, batchsize, datasetname)

test_accs.append((accs_0 + accs_0 + accs_0 + accs_0 + accs_0) / 5)
R_pre.append((pre1_0+pre1_0+pre1_0+pre1_0+pre1_0)/5)
R_rec.append((rec1_0+rec1_0+rec1_0+rec1_0+rec1_0)/5)
R_F1.append((F1_0 + F1_0 + F1_0 + F1_0 + F1_0) / 5)
NR_pre.append((pre2_0+pre2_0+pre2_0+pre2_0+pre2_0)/5)
NR_rec.append((rec2_0+rec2_0+rec2_0+rec2_0+rec2_0)/5)
NR_F1.append((F2_0 + F2_0 + F2_0 + F2_0 + F2_0) / 5)
result_str = "AVG_result: {:.4f} \n NR  pre: {:.4f}  rec: {:.4f}  F1: {:.4f} \n R pre: {:.4f}  rec: {:.4f}  F1: {:.4f}".format(sum(test_accs), sum(NR_pre), sum(NR_rec), sum(NR_F1),sum(R_pre), sum(R_rec),sum(R_F1))
#result_str = "AVG_result: {:.4f} \n NR  pre: {:.4f}  rec: {:.4f}  F1: {:.4f} \n R pre: {:.4f}  rec: {:.4f}  F1: {:.4f}".format(accs_0, pre1_0, rec1_0, F1_0,pre2_0, rec2_0,F2_0)
print(result_str)

with open('./{}_final_result.txt'.format(datasetname), 'a+') as f:
    f.write(time.strftime("%Y-%m-%d %H:%M:%S\n =======New Method=======\n", time.localtime()))
    f.write("\n")
    f.write(result_str+"\n")
    f.write(str(opt))
    f.write("###################################\n")