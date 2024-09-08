import torch
from torch.utils.data import Dataset
import pickle as pkl
from transformers import AutoTokenizer


def loadData(dataname, fold_x_train, fold_x_test):
    textPath = './autodl-fs/data/{}/new_textDic.pkl'.format(dataname)
    imgPath = './autodl-fs/data/{}/new_imgDic.pkl'.format(dataname)
    labelPath = './autodl-fs/data/{}/new_labelDic.pkl'.format(dataname)
    commPath = './autodl-fs/data/{}/new_commDic.pkl'.format(dataname)
    senPath = './autodl-fs/data/{}/new_senDic.pkl'.format(dataname)
    # genPath = './data/{}/new_genDic.pkl'.format(dataname)

    textDic = pkl.load(open(textPath,"rb"))
    commDic = pkl.load(open(commPath, "rb"))
    senDic = pkl.load(open(senPath, "rb"))
    imgDic = pkl.load(open(imgPath,"rb"))
    labelDic = pkl.load(open(labelPath,"rb"))
    # genlabelDic = pkl.load(open(genPath, "rb"))

    ids = list(imgDic.keys())
    new_fold_x_train = []

    for i in fold_x_train:
        if i in ids:
            new_fold_x_train.append(i)
    new_fold_x_test = []
    for j in fold_x_test:
        if j in ids:
            new_fold_x_test.append(j)
    new_fold_x_val = []

    # train_da = [i+'_1' for i in new_fold_x_train]
    # new_fold_x_train = new_fold_x_train + train_da
    print(len(new_fold_x_train), len(new_fold_x_test))

    print("loading train set", )
    traindata_list = RumorDataset(new_fold_x_train, textDic, senDic, commDic, imgDic, labelDic)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = test_RumorDataset(new_fold_x_test, textDic, senDic,commDic, imgDic, labelDic)

    print("test no:", len(testdata_list))

    return traindata_list, testdata_list

class RumorDataset(torch.utils.data.Dataset):
    def __init__(self, fold_x, textDic, senDic, commDic, imgDic, labelDic):
        self.fold_x = fold_x
        self.textDic = textDic
        self.senDic = senDic
        self.commDic = commDic
        self.imgDic = imgDic
        self.labelDic = labelDic

    def __getitem__(self, idx):
        id = self.fold_x[idx]
        label = self.labelDic[id]
        text = self.textDic[id]
        comm = torch.tensor(self.commDic[id]).squeeze(1)  #30 * 768
        sen = torch.tensor(self.senDic[id]) # 30 * 1
        img = self.imgDic[id]
        item1 = {key: torch.tensor(val) for key, val in text.items()}
        item1['img'] = img
        item1['labels'] = torch.tensor(label).long()
        item1['comm'] = comm.float()
        item1['sen'] = sen.float()
        #item1["gen_label"] = tokenizer("false", return_tensors="pt").input_ids if label==0 else tokenizer("true", return_tensors="pt").input_ids

        return item1

    def __len__(self):
        return len(self.fold_x)

class test_RumorDataset(torch.utils.data.Dataset):
    def __init__(self, fold_x, textDic, senDic, commDic, imgDic,labelDic):
        self.fold_x = fold_x
        self.textDic = textDic
        self.commDic = commDic
        self.senDic = senDic
        self.imgDic = imgDic
        self.labelDic = labelDic

    def __getitem__(self, idx):
        id = self.fold_x[idx]
        label = self.labelDic[id]
        text = self.textDic[id]
        comm = torch.tensor(self.commDic[id]).squeeze(1)
        sen = torch.tensor(self.senDic[id])
        img = self.imgDic[id]
        item1 = {key: torch.tensor(val) for key, val in text.items()}

        item1['img'] = img
        item1['comm'] = comm.float()
        item1['sen'] = sen.float()
        item1['labels'] = torch.tensor(label).long()
        # item1["gen_label"] = tokenizer("false", return_tensors="pt").input_ids if label == 0 else tokenizer("true",return_tensors="pt").input_ids
        # item['labels'] = torch.LongTensor(self.labels[idx])
        return item1

    def __len__(self):
        return len(self.fold_x)