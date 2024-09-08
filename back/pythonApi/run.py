import time
from flask import Flask, request, jsonify
import base64
import os
import json
from PIL import Image
import io

import torch

from preProcess import img_prase, reply_comm_prase, reply_sen_prase, source_text_prase
from MFModel import MF_Net,hard_fc

app = Flask(__name__)

model = torch.load('./model/MF_model/model.pth.tar',map_location=torch.device('cpu'))
model.eval()

def getData(source_text,reply_text,image,model):

    input_ids,attention_mask,token_type_ids = source_text_prase(source_text,'./model/bert-base-chinese')
    img1 = img_prase(image)
    comm = reply_comm_prase(source_text,reply_text,'./model/bert-base-chinese',30,400)
    sen = reply_sen_prase(reply_text,30,400)
    labels = torch.tensor([0])
    pred = model(input_ids,attention_mask,token_type_ids,img1,labels,comm,sen)
    pred = pred.tolist()
    if pred[0][1] > pred[0][0]:
        return 1,pred[0][1] * 100
    else:
        return 0,pred[0][0] * 100



def getJson(data):
    '''
    处理前端返回的json数据
    '''
    source_texts = []
    reply_texts = []
    images = []
    for item in data:
        # 检查字段是否存在，如果不存在则添加空字符串
        if 'source_text' not in item:
            item['source_text'] = "这是填充语句"
        if 'reply_text' not in item:
            item['reply_text'] = []
        source_texts.append(item['source_text'])
        reply_texts.append(item['reply_text'])
        if 'image' not in item:
            # 如果缺少图像字段，则创建一个空白图像
            blank_image = Image.new('RGB', (100, 100), (255, 255, 255))
            images.append(blank_image)
        else:
            if len(item['image']) == 0:
                blank_image = Image.new('RGB', (100, 100), (255, 255, 255))
                images.append(blank_image)
            else:
                base64_image_data = item['image']
                base64_image_data = data[0]['image'].split('base64,')[1]
                base64_image_data = bytes(base64_image_data,'utf-8')
                image_data = base64.b64decode(base64_image_data)
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                images.append(image)
    return source_texts[0],reply_texts[0],images[0]

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    data[0]['reply_text'] = data[0]['reply_text'][:5]
    source_text,reply_text,imageBase64 = getJson(data)
    # 在这里可以对源文本、回复文本和图片做进一步处理，比如保存到数据库或进行其他分析操作
    #model = torch.load('./model/MF_model/model.pth.tar',map_location=torch.device('cpu'))
    #model.eval()

    T1 = time.time()
    isRumor,rate = getData(source_text,reply_text,imageBase64,model)
    T2 = time.time()

    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    return jsonify([
        {
            "isRumor":isRumor,
            "rate":rate,
        }
    ]), 200

if __name__ == '__main__':
    app.run('127.0.0.1',port=5005,debug = True)

