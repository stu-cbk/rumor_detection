
import re
from transformers import BertTokenizerFast,BertModel
import torch
import json
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests

def clear(str):
    results = re.compile(r'http://[a-zA-Z0-9.?/&=:]*',re.S)
    s = results.sub("",str)
    new_s = re.sub("[^A-Za-z ]","",s)
    return new_s

def source_text_prase(source_text,bert_path):
    '''
    预处理source_text
    '''
    tokenizer = BertTokenizerFast.from_pretrained(bert_path)
    source_text = source_text.encode('UTF-8', 'ignore').decode('UTF-8')
    source_text = re.sub(r'http\S+', '', source_text)
    source_text = clear(source_text)

    #collate_fn(source_text,512,tokenizer,device)
    encoding = tokenizer([source_text],  # 直接调用PreTrainedTokenizerFast.__call__方法
                        is_split_into_words=True,
                        return_offsets_mapping=True,
                        max_length = 512,
                        padding= 'max_length', #'max_length',  #  True 暂为设置max_length，后期可作为超参数使用
                        truncation=True,
                        return_token_type_ids=True)
    return torch.tensor(encoding['input_ids']).view(1,-1),torch.tensor([encoding['attention_mask']]).view(1,-1),torch.tensor([encoding['token_type_ids']]).view(1,-1)

def img_prase(im):
    '''
    预处理img
    '''
    #im = Image.open(imagePath).convert('RGB')
    trans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    im1 = trans(im)  # 原图片
    #print(im1.shape)
    return im1.view(1,3,224,224)

# 获取token
def getToken():
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    App_Key = "S5a0AkR7L1av566SpWKLpFYf"
    Secret_Key = "CjcpG0ZirWZRWkAQ0DTyHwK0Jsd4pIux"
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id='+App_Key + '&client_secret=' + Secret_Key
    response = requests.get(host)

    if response.status_code == 200:
        info = json.loads(response.text)  # 将字符串转成字典
        access_token = info['access_token']  # 解析数据到access_token
        return access_token
    return ''

def get_emotion(inputText:str):
    '''
    情感倾向分析
    '''
    dic = {2:"positive",1:"neutral",0:"negetive"}
    token = getToken()
    # token = '24.00f63feb99c9147455aca360239082f7.2592000.1713188026.282335-56808387'
    url = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token={}'.format(token)

    header = {'Content-Type ': 'application/json'}
    body = {'text': inputText}
    requests.packages.urllib3.disable_warnings()
    res = requests.post(url=url, data=json.dumps(body), headers=header, verify=False)
    #  返回接口格式：
    # {
    #     "log_id": 7475291888689599393,    # 请求唯一标识码
    #     "text": "我最好看",               #
    #     "items": [
    #         {
    #             "positive_prob": 0.999976,   # 表示属于积极类别的概率 ，取值范围[0,1]
    #             "confidence": 0.999946,       # 表示分类的置信度，取值范围[0,1]
    #             "negative_prob": 2.42354e-05,  # 表示属于消极类别的概率，取值范围[0,1]
    #             "sentiment": 2             # 表示情感极性分类结果，0:负向，1:中性，2:正向
    #         }]
    # }

    if res.status_code == 200:
        info = json.loads(res.text)
        # print(info)
        if 'items' in info and len(info['items']) > 0:
            return info['items'][0]['sentiment']
            # if sentiment == 2:
            #     print(inputText+'  情感分析结果是:正向')
            # elif sentiment == 1:
            #     print(inputText + '  情感分析结果是:中性')
            # else:
            #     print(inputText + '  情感分析结果是:负向')
        else:
            return 1

def reply_comm_prase(source_text,reply_text:list,bert_path,max_comm_num = 30,max_reply_sen = 400):
    '''
    预处理评论的句子姿态信息
    '''
    model = BertModel.from_pretrained(bert_path)
    tokenizer = BertTokenizerFast.from_pretrained(bert_path)
    comm = []
    reply_text_len = len(reply_text)

    if reply_text_len >= max_comm_num:
        comm += reply_text[:max_comm_num]
    else:
        comm += reply_text + ['MASK'] * (max_comm_num - reply_text_len)

    source_text = clear(source_text).split()
    source_texts = []
    reply_texts = []
    #print(len(comm))
    for r in comm:
        reply = clear(r).strip().split()[:max_reply_sen]
        if not reply or len(reply) == 0:
            reply = ['MASK']
        source_texts.append(source_text)
        reply_texts.append(reply)
    #print(len(source_texts))

    encoding = tokenizer(source_texts,
                        reply_texts,
                        is_split_into_words=True,
                        return_offsets_mapping=True,
                        padding=True,  # 暂为设置max_length，后期可作为超参数使用
                        truncation=True,
                        return_token_type_ids=True)   
    input_ids = torch.tensor(encoding['input_ids'])
    s_output = model(input_ids)
    s_guide_out = s_output[0][:, 0, :]
    #print(s_guide_out.shape)
    return s_guide_out.view(1,30,768)

def reply_sen_prase(reply_text:list,max_comm_num = 30,max_reply_sen = 400):
    '''
    预处理评论的情感倾向
    '''

    if len(reply_text) >= max_comm_num:
        reply_text = reply_text[:max_comm_num]

    emotion = []
    for reply in reply_text:
        reply = clear(reply).strip().split()[:max_reply_sen]
        reply = str(reply)
        emo = get_emotion(reply)
        emotion.append(emo)
    
    if len(emotion) < max_comm_num:
        emotion = emotion + [1] * (max_comm_num - len(emotion))

    # print(len(emotion))
    return torch.tensor(emotion).float().view(1,30)
