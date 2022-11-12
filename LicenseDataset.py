# _*_coding:utf-8_*_
'''
数据库= n * 许可证
'''

import json
import logging
import os
import re
import pandas as pd
import pickle

from Term import Term
from License import License
import utils
from model.config import config as term_config



DIR = os.path.dirname(os.path.abspath(__file__))+'/'


class Licensedataset:
    def __init__(self, licenseList=None):
        self.licenseList = licenseList

        self.licenses = None # dict(name:text). 未经结构化的许可证数据库（原始的若干个许可证文本）
        self.sentBertIdsDataset = None # list的list。 （若干个句子的ids）（各个许可证的句子ids，总体再消重）对应roberta-base的。

        if self.licenseList is None:
            self.licenseList = []

    def printLicenseList(self):
        for ll in self.licenseList:
            print(ll.getName(), ll.getTermList())
        return


    def addLicense(self, license):
        self.licenseList.append(license)
        return


    def load_licenses_from_csv(self, nlp, ld, ner_model_ee5, re_args, re_model, ac_model):
        '''
        直接读取 已经结构化的许可证 数据库
        :return:
        '''

        df = pd.read_csv(DIR+"data/tldr-licenses-forSpdx.csv")
        # contentList = list(df.columns)[1:]

        for row in df.itertuples():
            # 每行是一个许可证
            i = len(self.licenseList)

            # 获取文本内容
            words, labs, entities_chunks = utils.get_entities(DIR + "data/termEntityTagging/" + str(i + 1) + '.txt', clean=False)
            text = ' '.join(words)
            ### 构造一个License对象
            li = License(name=row[1], text=text, matchedLnameList=[], textNeedTE=True)

            if os.path.exists(os.path.join(DIR, 'ld_save', li.name+'.json')):

                with open(os.path.join(DIR, 'ld_save', li.name+'.json'), 'r', encoding="utf-8") as fr:
                    liJSON = json.load(fr)
                    for tjson in liJSON:
                        tt = Term()
                        tt.setContent(tjson['content'])
                        tt.setAtti(tjson['atti'])
                        tt.setRecipient(tjson['recipient'])
                        tt.setCondInxs(tjson['condInxs'])
                        li.addTerm(tt)
                assert len(li.termList) == 23

            else:

                li.termExtraction(nlp, ld, ner_model_ee5, re_args, re_model, ac_model)
                with open(os.path.join(DIR, 'ld_save', li.name + '.json'), 'w', encoding="utf-8") as fw:
                    liJSON = []
                    for tt in li.termList:
                        tjson = {}
                        tjson['content'] = tt.content
                        tjson['atti'] = tt.atti
                        tjson['recipient'] = tt.recipient
                        tjson['condInxs'] = tt.condInxs
                        liJSON.append(tjson)
                    json.dump(liJSON, fw)




            # 覆盖atti
            for j, atti in enumerate(row[2:]):
                # 某许可证的一个条款with极性
                li.termList[j].setAtti(atti=atti)
                # 设置缺省认定值 （这里就都设成123 省的兼容性检测时不统一 导致bug）
                li.termList[j].set_absentAtti()
                # ### 更新self.termList
                # li.addTerm(tt)

            assert len(li.termList) == 23
            self.addLicense(li)

            print("load ld: ", i)

        ##self.printLicenseList() #### （海星 cond）
        return self.licenseList



    def give_termList_from_liname(self, name):
        for li in self.licenseList:
            kk = li.name.split('___')
            for k in kk:
                if k==name:
                    return li.termList
        print('【这个matchedLiName竟然在ld里面找不到对应的】,,,,,', name)

        # （记录一下）
        with open(os.path.join(DIR, 'gap_spdx_tldr.txt'), 'a', encoding="utf-8") as fw:
            fw.write(name + '\n')


        return []



    def read_licenses(self, dataDir):
        '''
        读取原始的若干个许可证文本；
        文本预处理；
        :return:
        '''
        licenses = {}
        for file in os.listdir(dataDir):
            with open(os.path.join(dataDir, file), 'r', encoding="utf-8")as fr:
                text = ' '.join([line.strip() for line in fr.readlines()])
            text = utils.cleanText(text)
            fr.close()
            # print(text)
            licenses[file[:-4]] = text
        self.licenses = licenses
        print('self.licenses', len(self.licenses))
        return self.licenses


    def generate_bert_ids_for_licenses(self,tokenizer, idsDir, max_seq_length):
        '''
        生成input_ids.h5，（是list的list）（若干个句子的ids）（各个许可证的句子ids，总体再消重）
        对应roberta-base的。
        '''


        ids = []
        for text in self.licenses.values():
            sentences = utils.sentences_split(text)
            for sent in sentences:
                sent = sent.strip().split(' ')[:max_seq_length] ###
                sent_ids = utils.generate_bert_ids_for_sentence(tokenizer=tokenizer,sentence=sent, fg=2)
                ids.append(sent_ids)
        # ids = list(set(ids))
        ids = utils.get_unique_lists_in_list(ids)
        print('ids', len(ids))
        self.sentBertIdsDataset = ids

        # 写文件


        import h5py
        f = h5py.File(idsDir, 'w')  # 创建一个h5文件，文件指针是f
        #f['data'] = str(ids)  # 将数据写入文件的主键data下面
        f.create_dataset(name='data', data=ids, dtype=int)
        f.close()


        return self.sentBertIdsDataset


    def generate_entity_mention_position_file(self, entity_mention_set, posDir):
        '''
        为了“mention融合成entity”，需要提前搜集该mention在数据库中(即self.sentBertIdsDataset)出现的所有句子 作为生成embedding的基础，
        一个mention有一个group，里面是若干个出现（在某句中的某位置）
        生成entity_pos.pkl
        （暂时 每个待预测许可证生成一个entity_mention_set，再生生成对应的一个pkl文件吧）
        '''
        # 初始化
        groups = {}
        for j in range(len(entity_mention_set)):
            groups[j] = [] # 一个group

        # 遍历self.sentBertIdsDataset，填充groups
        for i in range(len(self.sentBertIdsDataset)):
            sent_ids = self.sentBertIdsDataset[i]
            for j in range(len(entity_mention_set)):
                phrase_ids = entity_mention_set[j]

                sent_str = ' '.join([ str(a) for a in sent_ids])
                phra_str = ' '.join([ str(a) for a in phrase_ids])
                if sent_str.find(phra_str) > -1:
                    # （可能有多次出现在此句中）
                    #starts = [each.start() for each in re.finditer(phra_str, sent_str)] # 注意 空格 对于id-pos是多余的
                    starts = [sent_str[:each.start()].count(' ')+1-1 for each in re.finditer(phra_str, sent_str)]
                    ends = [start + len(phrase_ids) for start in starts] #### 左开右闭
                    spans = [(start, end) for start, end in zip(starts, ends)]
                    for sp in spans:
                        # 一次出现
                        cur_item = [i, sp[0], sp[1]]
                        groups[j].append(cur_item)
        print('groups', len(groups))
        # for j in range(len(entity_mention_set)):
        #     print(str(len(groups[j])))

        # 以二进制方式来存储,rb,wb,wrb,ab
        p = open(posDir, 'wb')
        # 将字典数据存储为一个pkl文件
        pickle.dump(groups, p)
        p.close()

        return groups










    def isNeedSatisfied(self,termList):
        '''
        判断本数据库中 是否存在满足此需求的许可证 【（准确符合这个需求）】
        输出license对象的列表
        :return:
        '''
        abled = []
        for ll in self.licenseList:
            if ll.isSatisNeed(termList):
                abled.append(ll)
        return abled

    def isNeedSatisfied_2(self,termlist_need_fromChildren, termlist_need_fromParent):
        '''
        判断本数据库中 是否存在满足此需求的许可证 【（满足这个范围要求）】
        输出license对象的列表
        :return:
        '''
        abled = []
        for ll in self.licenseList:
            if ll.isSatisNeed_2(termlist_need_fromChildren, termlist_need_fromParent):
                abled.append(ll)
                print(ll.name, ' '.join([str(k) for k in ll.printTermlist()]))
        return abled








'''
ld = Licensedataset()
ld.printLicenseList()

license = License(name="GYL")
license.addTerm(Term(content="Distribute",atti="cannot"))
ld.addLicense(license)
ld.printLicenseList()
'''

'''
ld = Licensedataset()
ld.load_licenses_from_csv()
ld.printLicenseList()
'''


# df = pd.read_csv(DIR+"data/tldr-licenses-forSpdx.csv")
# print(list(df.columns)[1:])



