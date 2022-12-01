# _*_coding:utf-8_*_
import json
import logging
import os
import re
from itertools import product
import shutil

import utils


from RE import re_predict

from EE5.LocateTerms import ner_predict

'''

'''

DIR = os.path.dirname(os.path.abspath(__file__))+'/'


class TermRelated:
    def __init__(self, sentence=None, action_idxs=None, action=None, action_j=None, action_atti=None):
        self.Sentence = sentence # strings
        self.Action_idxs = action_idxs
        self.Action = action # strings
        self.action_j = action_j # int (0-22)
        self.action_atti = action_atti # str
        ##
        self.Performer = 'The licensor ' # str
        self.Recipient = 'this work ' # str
        self.Attitude = 'can ' # str
        self.Condition = [] # list[ dict{"action":str, "performer":str, "recipient":str, "attitude":str  } ]





    def predict_allEntityExtraction(self, ner_model_ee5):
        '''
        【输入self.Sentence
        调用已经训练好的模型，识别出所有possible的实体们
        【得到所有实体的对应的words, labs, entities_chunks
        '''
        EEdir = DIR+'EE5/LocateTerms/'

        # (self.Sentence已经在getOOO和getItsSequence都做过清洗了，直接OOO就行)

        # # 放入EE5的测试数据文件夹
        # utils.write_BIO_file(self.Sentence.split(' '), ['O']*len(self.Sentence.split(' ')),
        #                      os.path.join(EEdir, 'data/test', 'oneSentenceFromTR.txt'))
        #
        # # 进行预测
        # ner_predict.main(model=ner_model_ee5)

        # print(self.Sentence)
        # print(self.Sentence.split(' '))

        ''' （先用旧的调通代码 等lly的弄好再换进来） '''
        # 放入EE5的测试数据文件夹
        utils.write_BIO_file([self.Sentence.split(' ')], [['O']*len(self.Sentence.split(' '))],
                             os.path.join(EEdir, 'data/test', 'oneSentenceFromTR.txt'))

        # 进行预测
        ner_predict.main(model=ner_model_ee5)

        # 从NER结果（test-pre/） 得到self的words, labs, entities_chunks
        words, labs, entities_chunks = utils.get_entities(
            os.path.join(EEdir, 'data/test-pre/', 'oneSentenceFromTR.txt'), clean=False)

        assert len(words)==len(labs)
        # print(len(words), len(self.Sentence.split(' ')))
        assert len(words)==len(self.Sentence.split(' ')) # （因为要保证action的位置依旧 在EE的过程中没被弄乱）

        ##
        for d in [
            DIR + 'EE5/LocateTerms/data/test/',
            DIR + 'EE5/LocateTerms/data/test-pre/',
        ]:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                    os.mkdir(d)
                except Exception as e:
                    print(e, d)
                    continue

        return words, labs, entities_chunks


    def prepare_data_fromEE_toREpredict_0(self, words, labs, entities_chunks):
        '''
        输入: EE5的输出数据
        输出：RE的输入数据
        '''
        # id2rel = utils.get_id2rel(filename=r'./rel2id-relation.json')
        dataList = []

        possible_CA_list = []

        for i, entity_chunk in enumerate(entities_chunks):
            # 对每一个实体

            ## 看看是否出现条件
            et_type = entity_chunk[0]
            if et_type=='ConditionalAction':
                possible_CA_list.append(i)

            ### （组装）
            sp_dict = {}
            sp_dict["token"] = words
            sp_dict["h"] = {}
            sp_dict["h"]["name"] = self.Action # 动作
            sp_dict["h"]["pos"] = list(self.Action_idxs)
            sp_dict["t"] = {}
            sp_dict["t"]["name"] = ' '.join(words[entity_chunk[1]:entity_chunk[2]]) # 另外一个实体
            sp_dict["t"]["pos"] = entity_chunk[1:3]
            sp_dict["relation"] = 'UNKNOWN'
            dataList.append(sp_dict)

        #assert len(dataList)==len(entities_chunks)

        # （若存在条件(存在条件动作)：把它也和其他实体组合一遍（除了自己））
        if possible_CA_list:

            for cai in possible_CA_list: # （一般也就最多一两个吧）
                for i, entity_chunk in enumerate(entities_chunks):
                    # 对每一个实体
                    if i==cai:
                        continue
                    ### （组装）
                    sp_dict = {}
                    sp_dict["token"] = words
                    sp_dict["h"] = {}
                    sp_dict["h"]["name"] = ' '.join(words[entities_chunks[cai][1]:entities_chunks[cai][2]])  # 条件动作
                    sp_dict["h"]["pos"] = entities_chunks[cai][1:3]
                    sp_dict["t"] = {}
                    sp_dict["t"]["name"] = ' '.join(words[entity_chunk[1]:entity_chunk[2]])  # 另外一个实体
                    sp_dict["t"]["pos"] = entity_chunk[1:3]
                    sp_dict["relation"] = 'UNKNOWN'
                    dataList.append(sp_dict)


        return dataList


    def compose_one_RE_sample(self, words, etc_h, etc_t):
        sp_dict = {}
        sp_dict["token"] = words
        sp_dict["h"] = {}
        sp_dict["h"]["name"] = ' '.join(words[etc_h[1]:etc_h[2]])
        sp_dict["h"]["pos"] = etc_h[1:3]
        sp_dict["t"] = {}
        sp_dict["t"]["name"] = ' '.join(words[etc_t[1]:etc_t[2]])
        sp_dict["t"]["pos"] = etc_t[1:3]
        sp_dict["relation"] = 'Other'
        return sp_dict

    def prepare_data_fromEE_toREpredict(self, words, labs, entities_chunks):
        '''
        输入: EE5的输出数据
        输出：RE的输入数据
        '''
        dataList = []

        # 找到所有的 各类型实体
        all_action_list = []
        all_recipient_list = []
        all_attitude_list = []
        all_condition_list = []
        for i, entity_chunk in enumerate(entities_chunks):
            # 对每一个实体
            et_type = entity_chunk[0]
            if et_type=='Action':
                all_action_list.append(i)
            elif et_type=='Recipient':
                all_recipient_list.append(i)
            elif et_type=='Attitude':
                all_attitude_list.append(i)
            elif et_type=='Condition':
                all_condition_list.append(i)

        # 组装：动作和对象
        for k in all_action_list:
            for t in all_recipient_list:
                sp_dict = self.compose_one_RE_sample(words, entities_chunks[k], entities_chunks[t])
                dataList.append(sp_dict)
        # 组装：动作和态度
        for k in all_action_list:
            for t in all_attitude_list:
                sp_dict = self.compose_one_RE_sample(words, entities_chunks[k], entities_chunks[t])
                dataList.append(sp_dict)
        # 组装：动作和条件
        for k in all_action_list:
            for t in all_condition_list:
                sp_dict = self.compose_one_RE_sample(words, entities_chunks[k], entities_chunks[t])
                dataList.append(sp_dict)
        # 组装：条件和动作
        for k in all_condition_list:
            for t in all_action_list:
                sp_dict = self.compose_one_RE_sample(words, entities_chunks[k], entities_chunks[t])
                dataList.append(sp_dict)

        return dataList




    def predict_relationExtraction(self, dataList, re_args, re_model):
        '''
        调用已经训练好的模型，【目的是预测已有action和所有entity的关系类别】，（模型输出的是每一对实体的关系类别）
        经过检查和过滤，（其实EE5之后已经有对实体类型的推测了，但经过关系分类，再一次矫正and去掉关系概率低的搭配，）
        【效果：填充进self.Performer，self.Recipient，self.Attitude，self.Condition）
        '''
        REdir = DIR+'RE/'

        # 放入RE的测试数据文件夹
        utils.write_RE_file(dataList, os.path.join(REdir, 'dataset/ossl2', 'test.txt'))

        ### 进行预测
        # （那些参数已经都变成了默认参数 不用另外再给了。）
        test_pre_logits, preds = re_predict.predict_re(args=re_args, lit_model=re_model)
        # print(len(preds),len(dataList))
        # assert len(preds)==len(dataList)
        if len(preds)!=len(dataList):
            print('!!!!! len(preds)!=len(dataList) from one sent', len(preds),len(dataList))
            return [], []

        # 暂时用preds给把dataList里面的label补全
        id2rel = utils.get_id2rel(filename=os.path.join(REdir, 'dataset/ossl2', 'rel2id.json'))
        dataList_final = []
        for i, sp_dict in enumerate(dataList):
            sp_dict["relation"] = id2rel[preds[i]]
            dataList_final.append(sp_dict)


        return test_pre_logits, dataList_final





