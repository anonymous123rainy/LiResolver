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


# import sys
# sys.path.append(os.path.realpath("."))
# import utils

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'




# 定义参数
batch_size = 4
epoches = 10 ###
model = "bert-base-uncased"
hidden_size = 768
n_class = 23
maxlen = 10



# word_list = ' '.join(sentences).split()
# word_list = list(set(word_list))
# word_dict = {w: i for i, w in enumerate(word_list)}
# num_dict = {i: w for w, i in word_dict.items()}
# vocab_size = len(word_list)

# 将数据构造成bert的输入格式
# inputs_ids: token的字典编码
# attention_mask:长度与inputs_ids一致，真实长度的位置填充1，padding位置填充0
# token_type_ids: 第一个句子填充0，第二个句子句子填充1
class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels=None, with_labels=True,):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.with_labels = with_labels
        self.sentences = sentences
        self.labels = labels
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent = self.sentences[index]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,       # Truncate to max_length
                                      max_length=maxlen,
                                      return_tensors='pt')  # Return torch.Tensor objects

        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.labels[index]
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids



# model
class BertClassify(nn.Module):
    def __init__(self):
        super(BertClassify, self).__init__()
        self.bert = AutoModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.linear = nn.Linear(hidden_size, n_class) # 直接用cls向量接全连接层分类
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # 返回一个output字典
        # 用最后一层cls向量做分类
        # outputs.pooler_output: [bs, hidden_size]
        logits = self.linear(self.dropout(outputs.pooler_output))

        return logits



def get_chunks2(labs):
    # 得到实体列表 [('X',3,4), (), () ...]
    # (左闭右开)
    TMP = []
    tmp = []
    for i in range(len(labs)):
        la = labs[i]
        if la.split('-')[0]=='B' or la.split('-')[0]=='I':
            if i==0 or labs[i-1]=='O' or labs[i-1].split('-')[1] != la.split('-')[1]:
                tmp.append(la.split('-')[1])
                tmp.append(i)
                tmp.append(i + 1)
            else:
                tmp[2] += 1
            if i==len(labs)-1 or labs[i+1]=='O' or labs[i+1].split('-')[1] != la.split('-')[1]:
                tmp2 = tuple(tmp)
                TMP.append(tmp2)
                tmp.clear()
    return TMP

def get_entities(filename, clean=True):
    '''
    :param filename: 读取NER-BIO形式的文本
    :return: words, labs, entities_chunks

    （要去除一下噪音字符）
    '''
    words = []
    labs = []
    with open(filename, 'r', encoding="utf-8")as fr:
        for line in fr.readlines():
            if line.strip():
                line = line.strip()

                assert len(line.split(' ')) == 2

                word = line.split(' ')[0].strip()

                if not word:
                    continue

                words.append(word)
                labs.append(line.split(' ')[1])

    entities_chunks = get_chunks2(labs)
    return words, labs, entities_chunks




# 训练数据
sentences = []
labels = []

'''

'''
dataDir = DIR + "data/termEntityTagging/"
for file in os.listdir(dataDir):
    words, labs, entities_chunks = get_entities(os.path.join(dataDir, file), clean=False)
    for ck in entities_chunks:
        text = ' '.join(words[ck[1]:ck[2]])
        tag = int(ck[0]) # 0-22
        sentences.append(text)
        labels.append(tag)
print('sentences: ', len(sentences))
print(sentences[:5])
print(labels[:5])

train = Data.DataLoader(dataset=MyDataset(sentences, labels), batch_size=batch_size, shuffle=True, num_workers=1)


bc = BertClassify().to(device)

optimizer = optim.Adam(bc.parameters(), lr=1e-3, weight_decay=1e-2)
loss_fn = nn.CrossEntropyLoss()

# train

score_history = []
best_score = 0

sum_loss = 0
total_step = len(train)

for epoch in range(epoches):
    predictions = []
    alllabels = []

    for i, batch in enumerate(train):
        optimizer.zero_grad()
        batch = tuple(p.to(device) for p in batch)
        pred = bc([batch[0], batch[1], batch[2]])


        #
        test_pre_logits = pred.detach().cpu().numpy()  # (n, 23)
        predLabels = np.argmax(test_pre_logits, axis=-1)  # 预测的标签 # (n,) [ 4 12 13 13 13 13 14 13 13 13]
        predictions.extend(predLabels)
        alllabels.extend(batch[3].detach().cpu().numpy())
        # print('predLabels:', predLabels)
        # print('trueLabels:', batch[3], batch[3].detach().cpu().numpy())

        loss = loss_fn(pred, batch[3])
        sum_loss += loss.item()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('[{}|{}] step:{}/{} loss:{:.4f}'.format(epoch+1, epoches, i+1, total_step, loss.item()))
    train_curve.append(sum_loss)
    sum_loss = 0

    #
    assert len(predictions)==len(alllabels)
    score_micro = f1_score(predictions, alllabels, labels=list(range(23)), average="micro")
    score_macro = f1_score(predictions, alllabels, labels=list(range(23)), average="macro")
    score_weigh = f1_score(predictions, alllabels, labels=list(range(23)), average="weighted")
    print('score_micro: ', score_micro)
    print('score_macro: ', score_macro)
    print('score_weigh: ', score_weigh)
    score_history.append(list([score_micro,score_macro,score_weigh]))

    #
    if score_weigh > best_score:
        best_score = score_weigh
        torch.save(bc.state_dict(), "./AC.pth")





print('score_history: ')
for i, s in enumerate(score_history):
    print(i, s)





# loss曲线
pd.DataFrame(train_curve).plot()