# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import transformers
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


train_curve = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


from .simpleTextClassifier import MyDataset, BertClassify

# 定义参数
batch_size = 4
epoches = 5 ###
model = "bert-base-uncased"
hidden_size = 768
n_class = 23
maxlen = 10

bc = BertClassify().to(device)


# load trained model
best_model_path = "./AC.pth"
bc.load_state_dict(torch.load(best_model_path)["state_dict"])



# predict
bc.eval()
with torch.no_grad():
    test_text = ['state changes']
    test = MyDataset(test_text, labels=None, with_labels=False)
    x = test.__getitem__(0)
    x = tuple(p.unsqueeze(0).to(device) for p in x)
    pred = bc([x[0], x[1], x[2]])
    print(pred)
    pred = pred.data.max(dim=1, keepdim=True)[1]
    print(pred[0][0])

