import torch
import torch.nn as nn
# from ResNet import resnet50

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