import torch
from PIL import Image
import argparse
from transformers import BertModel,BertTokenizerFast
from visModel import vgg
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cpu')

class VGG(nn.Module):
    """
    定义分类网络结构
    """
    def __init__(self, features, num_classes=1000, init_weights=True):#初始化函数，features是def make——features生成的提取特征网络结构，所需要分类的类别个数，是否权重初始化
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(    #三层全连接层
            nn.Dropout(p=0.5),#减少过拟合 以50%的概率随机失活神经元
            nn.Linear(512*7*7, 4096),#2048是节点个数
            nn.ReLU(True),#设置激活函数
            nn.Dropout(p=0.5),#随机失活
            nn.Linear(4096, 1024),#第二层全联接层  他的输入是上一层的输出节点个数
            nn.ReLU(True),
            nn.Dropout(p=0.3),  # 随机失活
            nn.Linear(1024, 768),  # 第二层全联接层  他的输入是上一层的输出节点个数
            # nn.Linear(2048, num_classes)#classnum是最终需要分类的类别个数
        )
        if init_weights:#初始化权重函数
            self._initialize_weights()

    def forward(self, x):
        """
        正向传播
        :param x:
        :return:
        """
        # N x 3 x 224 x 224
        x = self.features(x)#将数据传入到特征提取网络 得到输出x
        # N x 512 x 7 x 7
        #TODO：这里返回一个7*7*512的
        jj = x
        x = torch.flatten(x, start_dim=1)#将数据进行zhan平处理
        # N x 512*7*7
        x = self.classifier(x)#将特征矩阵输入到分类矩阵
        return x,jj

    def _initialize_weights(self):
        """
        遍历网络的每一层，当前层为卷积层 就用xavier来初始化，如果有偏执，默认为0，如果为全链接层，也用xavier初始化
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)#将偏置为0
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    """
    定义特征提取结构
    :param cfg:
    :return: 特征提取网络
    """
    layers = []
    in_channels = 3 #RGB彩色图片是3通道
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]#配置元素是M说明是最大池化层 就创建一个maxpooling
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v #这一层的输出深度是下一层的输入
    return nn.Sequential(*layers)# *是非关键字参数 通过非关键字参数输入进去 创建网络


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    #对应11层网络的结构，数字代表卷积核个数，M是赤化层结构（maxpooling）
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg19", **kwargs):
    """
    实例化vgg网络，
    :param model_name: 要搞哪个网络
    :param kwargs: 可变长度的字典变量。分类的类别个数以及是否初始化权重的布尔变量
    :return:
    """
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model

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

class MF_Net(torch.nn.Module):
    def __init__(self, opt):
        super(MF_Net, self).__init__()
        self.bert = torch.nn.DataParallel(BertModel.from_pretrained(opt.bert_path))
        self.vgg = torch.nn.DataParallel(vgg(model_name="vgg19"))
        #self.vgg = vgg(model_name="vgg19")
        # self.vgg = vgg(model_name="vgg19")
        # self.fc = torch.nn.Linear(2*opt.out_feats, 2)
        # self.pca = PCA(n_components=256)
        # self.embedding_dim = nn.Linear(768, opt.d_model)
        # 增加一个多头注意力模块
        # self.self_attn_layers_list = nn.ModuleList(
        #         [SelfAttentionSubEncoder(opt.n_head, opt.d_model, opt.d_qkv, opt.dropout_mha) for _ in range(opt.num_layers_mha)])
        self.lstm1 = nn.LSTM(768, 384, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(1, 384, batch_first=True, bidirectional=True)

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
            #print('cl_loss出现了nan值')
            cl_loss = torch.tensor(0)
        return cl_loss

    def forward(self, input_ids1, attention_mask1, token_type_ids1, img1, labels1, comm, sen):
        print("intput_ids: ",input_ids1.shape)
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
        #y = labels1  # y1是原数据的标签，y2是增强数据的标签
        #cl_loss1 = self.get_clloss(opt.T1, x, y)

        h0, c0 = self.init_hidden(comm.size(0), 384)
        output_1, (h, c) = self.lstm1(comm, (h0, c0))
        # print("sen.shape: ", sen.shape)
        sen = sen.unsqueeze(2)
        h0_2, c0_2 = self.init_hidden(sen.size(0), 384)
        output_2, (h_2, c_2) = self.lstm2(sen, (h0_2, c0_2))
        #new_h,new_h_1 = h[0,:,:],h[-1,:,:]
        output_1 = output_1[:,-1,:].squeeze(1)

        output_2 = output_2[:,-1,:].squeeze(1)
        Z = torch.cat((output_1, output_2), 1)  # 两种表征拼接
        Z = self.hard_fc2(Z)
        #cl_loss2 = self.get_clloss(opt.T2,Z,y)
        x = torch.cat((x, Z), 1)
        # print("x: ",x.shape)
        x = self.fc(x)
        #x = F.log_softmax(x, dim=1)
        #cl_loss = cl_loss1 + 0.1 * cl_loss2
        #print("cl_loss1: ",cl_loss1.item())
        #print("cl_loss2: ",cl_loss2.item())
        return F.softmax(x,dim=1)

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', default='./bert-base-chinese',)
    parser.add_argument('--out_feats', default=768*2)
    parser.add_argument('--T1', default=0.7)
    parser.add_argument('--T2', default=0.4)
    opt = parser.parse_args()
    return opt

opt = parse_option()