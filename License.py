# _*_coding:utf-8_*_
'''
一个许可证 = n * 条款
'''

import json
import logging
import os
import re
import pandas as pd
import shutil


from Term import Term
import utils
from TermRelated import TermRelated
from AC import shortTextClassification

from model.PreprocessData import cleanData_intoTestDir
from model.LocateTerms import ner_predict
from model.DetermAtti import get_treeAtti
from model.config import config as term_config


DIR = os.path.dirname(os.path.abspath(__file__))+'/'


class License:
    def __init__(self, name=None, termList=None, text=None, textNeedTE=None, matchedLnameList=None):
        '''
        过程中的被处理形式：期待是termList.
        :param name:
        :param termList:
        :param text:
        '''
        self.name = name
        self.termList = termList # termExtraction
        self.text = text # text. 经过条款提取进入termList

        self.textNeedTE = textNeedTE ##
        self.matchedLnameList = matchedLnameList ##

        self.entity_mention_set = None

        if self.termList is None:
            self.termList = []

        # (条款细节抽取的相关)
        self.words = None
        self.labs = None
        self.entities_chunks = None
        self.jj_etChunkInx = None ##
        ##
        self.termRelatedList = None # 来源：extract_termRelated()
        # List[ TermRelated(Object) ]

    def printTermlist(self, base_termlist=None):
        if base_termlist:
            attiList = [term_config['attiType_label'][tt.atti] for tt in base_termlist]
        else:
            attiList = [term_config['attiType_label'][tt.atti] for tt in self.termList]
        return attiList


    def termExtraction(self, nlp, ld, ner_model_ee5, re_args, re_model, ac_model):
        '''
        由self.text，进行条款提取；self.name当做data文件夹下的文件名
        填充其self.termList

        【这里的所有都只涉及到一个许可证(每次用NER预测一个)（不会被fname一样而影响）】

        【tree里的text一定要去检测CPS，有可能进行条款提取（根据标志位情况），所有ref的都放matchedLnameList去直接找label基础】
        '''

        ## 把matchedLnameList对应的label结果拿过来
        matchedLnameList = list(set(self.matchedLnameList))
        for mathedLiname in matchedLnameList:
            base_termlist = ld.give_termList_from_liname(mathedLiname)
            if base_termlist:
                self.setTermList(base_termlist)
                print('base_termlist', mathedLiname, ' '.join([str(k) for k in self.printTermlist(base_termlist=base_termlist)]))

        ''' 进行条款提取 '''
        print('self.textNeedTE:', self.textNeedTE)
        if self.textNeedTE:

            # # 预处理
            # with open(DIR + 'model/data/' + self.name + '.txt', 'w', encoding="utf-8") as fw:
            #     fw.write(self.text)
            # fw.close()
            # # 主体步骤
            # cleanData_intoTestDir.main()
            # ner_predict.main(model=ner_model)
            # _ = get_treeAtti.main(nlp=nlp)


            ''' 对于这篇文本 '''
            tmpTermList = []

            text = utils.cleanText(self.text)

            sentsList = utils.sentences_split(text)
            for sent in sentsList:
                ''' （按顺序）对每一个句子 '''

                ## 创建TermRelated对象 初始化
                tr = TermRelated(sentence=sent.strip(), )
                # （预测）实体识别
                words, labs, entities_chunks = tr.predict_allEntityExtraction(ner_model_ee5)
                # (ee->re 整理格式)
                dataList = tr.prepare_data_fromEE_toREpredict(words, labs, entities_chunks)
                if not dataList:
                    continue
                # （预测）关系识别
                test_pre_logits, dataList_final = tr.predict_relationExtraction(dataList, re_args, re_model)
                if not dataList_final:
                    continue

                # （（。。，如果EE准确率太低影响到整体效果，就在这里用test_pre_logits进行过滤筛选,得到新的dataList_final））
                ##

                ## （可能的条件后动作）
                ConditionalActionList = []
                for sp_dict in dataList_final:
                    if sp_dict["relation"] == "Condition-Action(e1,e2)" \
                            and utils.get_type_from_etcPos(entities_chunks, sp_dict["t"]["pos"])=='Action':
                        ConditionalActionList.append(sp_dict["t"]["pos"])
                condInx_jj = {}


                ## （所有动作）
                actionList = []
                # for i, entity_chunk in enumerate(entities_chunks):
                #     et_type = entity_chunk[0]
                #     if et_type == 'Action':
                #         #actionList.append(str(entity_chunk[1])+' '+str(entity_chunk[2]))
                #         actionList.append(entity_chunk[1:3])
                # 先放条件后动作 然后再条件前动作
                for i, entity_chunk in enumerate(entities_chunks):
                    et_type = entity_chunk[0]
                    if et_type == 'Action' and entity_chunk[1:3] in ConditionalActionList: #
                        actionList.append(entity_chunk[1:3])
                for i, entity_chunk in enumerate(entities_chunks):
                    et_type = entity_chunk[0]
                    if et_type == 'Action' and entity_chunk[1:3] not in ConditionalActionList: #
                        actionList.append(entity_chunk[1:3])

                # print('最初', ConditionalActionList)

                for at in actionList:
                    ''' 对每一个出现的action '''

                    ####
                    # 预备一个term对象（不一定会被消费）
                    tt = Term()
                    # 其对象
                    for sp_dict in dataList_final:
                        if sp_dict["h"]["pos"]==at and sp_dict["relation"]=="Action-Recipient(e1,e2)":
                            tt.setRecipient(recipient=sp_dict["t"]["name"])
                            break
                    if not tt.recipient:
                        tt.setRecipient(recipient="")
                    # 其动作(type)
                    termStr = ' '.join(words[at[0]:at[1]])+' '+tt.recipient
                    content_id = shortTextClassification.predict(text=termStr, ac_model=ac_model) ##(23分类)
                    content = term_config['term_list'][content_id]
                    tt.setContent(content=content)
                    # 其态度(type)
                    attilist = []
                    for sp_dict in dataList_final:
                        if sp_dict["h"]["pos"]==at and sp_dict["relation"]=="Action-Attitude(e1,e2)":
                            attilist.append(sp_dict["t"]["name"])
                    attiLabel = get_treeAtti.getAtti(attilist=attilist)
                    atti = term_config['attiLabel_type'][attiLabel]
                    tt.setAtti(atti=atti)
                    # 其条件(id-list)
                    if at in ConditionalActionList:
                        if tt.content not in [tm.content for tm in tmpTermList]:
                            ### tt可以被add
                            tt.setCondInxs(condInxs=[])
                            tmpTermList.append(tt) #####
                            condInx_jj[str(at[0])+' '+str(at[1])] = utils.get_type2id()[tt.content]
                            # print('添加为', condInx_jj)
                        else:
                            # 不能add那就顺便也从ConditionalActionList中除去
                            inx = ConditionalActionList.index(at)
                            ConditionalActionList.pop(inx)
                            # print(ConditionalActionList)

                    else:
                        if tt.content not in [tm.content for tm in tmpTermList]:
                            ### tt可以被add
                            # print(condInx_jj)
                            tt.setCondInxs(condInxs=[condInx_jj[str(ct[0])+' '+str(ct[1])] for ct in ConditionalActionList])
                            tmpTermList.append(tt) #####

            ''' (解析结束) '''
            print('len(tmpTermList):',len(tmpTermList))
            assert len(tmpTermList)<=23
            for j in range(23):
                content = term_config['term_list'][j]
                #if content in [tm.content for tm in tmpTermList]:

                if self.existsTerm(content=content): ## 已有base

                    #if tt.isMentioned(): # 1/2/3
                    if content in [tm.content for tm in tmpTermList]:
                        self.updateTerm(tmpTermList[[tm.content for tm in tmpTermList].index(content)]) ### 覆盖上去
                        #print('     updateTerm:', tmpTermList[[tm.content for tm in tmpTermList].index(content)].content, tmpTermList[[tm.content for tm in tmpTermList].index(content)].atti, '【from text：】', self.text)

                else:

                    if content in [tm.content for tm in tmpTermList]:
                        self.addTerm(tmpTermList[[tm.content for tm in tmpTermList].index(content)])  ###
                    else:
                        tt = Term(content=content)
                        tt.set_all_default()
                        self.addTerm(tt) ###
        print('len(self.termList):',len(self.termList))
        assert len(self.termList)==23

        return





    def getName(self):
        return self.name

    def getTermList(self):
        # return self.termList
        tmp = []
        for tt in self.termList:
            tmp.append(tt.get())
        return tmp

    def setTermList(self, termList):
        self.termList = termList
        return


    def addTerm(self, term):
        self.termList.append(term)
        return

    def updateTerm(self, tt):
        for term in self.termList:
            if term.content == tt.content:
                term.atti = tt.atti
        return

    def existsTerm(self, content):
        for term in self.termList:
            if term.content == content:
                return True
        return False


    def isSatisNeed(self, termList):
        '''
        给定需求，判断此license对象是否满足. 【（准确符合这个需求）】
        （满足给定的条款集合即可，其他多余的条款不管）
        :param termList:
        :return:
        '''
        for tn in termList:
            fg = False
            for term in self.termList:
                if term.content == tn.content and term.atti == tn.atti:
                    fg = True
                    break
            if not fg:
                return False
        return True


    def isSatisNeed_2(self, termlist_need_fromChildren, termlist_need_fromParent):
        '''
        给定需求，判断此license对象是否满足. 【（满足这个范围要求）】
        '''
        for j in range(23):

            if not termlist_need_fromParent:
                if not self.termList[j].isMoreStrict(termlist_need_fromChildren[j], self.termList, termlist_need_fromChildren):
                    return False
            else:
                if not (self.termList[j].isMoreStrict(termlist_need_fromChildren[j], self.termList, termlist_need_fromChildren)
                        and termlist_need_fromParent[j].isMoreStrict(self.termList[j], termlist_need_fromParent, self.termList)):
                    return False
                # # [遇到“父节点没权限时才考虑上层需求”]
                # if not self.termList[j].isMoreStrict(termlist_need_fromChildren[j], self.termList, termlist_need_fromChildren):
                #     return False

        return True






'''
license = License(name="GYL")
license.addTerm(Term(content="Distribute",atti="cannot"))
license.addTerm(Term(content="Distribute",atti="can"))
license.addTerm(Term("Modify","cannot"))
print(license.getTermList())
'''