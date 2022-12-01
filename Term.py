# _*_coding:utf-8_*_
import json
import logging
import os
import re
from itertools import product

from model.config import config as term_config
import utils

'''
'''
class Term:
    def __init__(self, content=None, atti=None, condInxs=None, recipient=None):
        self.content = content
        self.atti = atti

        self.condInxs = condInxs
        if self.condInxs is None:
            self.condInxs = []

        self.recipient = recipient
        if self.recipient is None:
            self.recipient = ""


    def composeOneSent(self,termlist):
        '''
        用这些被解构出的属性，组装出一条自然语言文本.
        :return:
        '''
        sent = ''
        sent += "This license "
        sent += 'claims that you '
        sent += self.atti +' '
        sent += self.content +' '
        if self.recipient:
            sent += 'for '
            sent += self.recipient

        if self.condInxs:
            sent += ', provided that : '
            for i, condInx in enumerate(self.condInxs):

                sent += 'you '
                sent += termlist[condInx].atti +' '
                sent += termlist[condInx].content +' '
                if termlist[condInx].recipient:
                    sent += 'for '
                    sent += termlist[condInx].recipient + ' '

                if i<len(self.condInxs)-1:
                    sent += 'and '

        sent += '.'

        return sent


    def get(self):
        return self.content, self.atti, self.condInxs

    def getAtti(self):
        return self.atti

    def set(self, content=None, atti=None):
        if content:
            self.content = content
        if atti:
            self.atti = atti
        return


    def setContent(self, content=None):
        if content:
            self.content = content
        return
    def setAtti(self, atti=None):
        if atti:
            self.atti = atti
        return
    def setRecipient(self, recipient=None):
        if recipient:
            self.recipient = recipient
        return
    def setCondInxs(self, condInxs=None):
        if condInxs:
            self.condInxs = condInxs
        return

    def set_all_default(self):
        # self.content = content
        self.atti = term_config['attiLabel_type'][0]
        self.set_absentAtti()
        self.condInxs = []
        self.recipient = ""
        return



    def set_absentAtti(self):
        '''
        权利cannot，义务can
        无返回值。直接修改自己。
        '''
        if self.atti==term_config['attiLabel_type'][0]:

            termId = term_config['term_list'].index(self.content)
            attiLabel = term_config['absentAtti'][termId]
            absentAtti = term_config['attiLabel_type'][attiLabel]
            self.atti = absentAtti

        return

    def isMentioned(self):
        if self.atti == term_config['attiLabel_type'][0]:
            return False
        return True


    def isconflict(self, termB):
        '''
        是否存在不一致(冲突)
        '''
        if self.content == termB.content and self.atti != termB.atti: #
            return True
        return False

    def isconflict2(self):
        if self.atti == term_config['attiLabel_type'][4]:
            return True
        return False


    def isSameContent(self, termB):
        if self.content == termB.content : #
            return True
        return False

    def isTwoOccurConflict(self, termB):
        '''
        self比termB 冲突（CL和CL的那种）
        （前置情况：都是1/2/3.）
        '''
        la1 = term_config['attiType_label'][self.atti]
        la2 = term_config['attiType_label'][termB.atti]
        la3 = term_config['atti_moreStrictTable'][la1 - 1][la2 - 1]
        # print(la1,la2,la3)
        if la3 == 4:
            return True
        else:
            return False


    def isMoreStrict(self, termB, termlistA, termlistB):
        '''
        self比termB 一样or更加严格
        （前置情况：他俩已经都非confilct了，都是1/2/3）

        '''

        if not termlistA or not termlistB:
            # （第二层进来的）

            if termB.atti == term_config['attiLabel_type'][4]:
                return False

            la1 = term_config['attiType_label'][self.atti]
            la2 = term_config['attiType_label'][termB.atti]
            la3 = term_config['atti_moreStrictTable'][la1 - 1][la2 - 1]
            # print(la1,la2,la3)
            if la3 == la1:
                return True
                # if utils.clean_recipientWords(self.recipient) == utils.clean_recipientWords(termB.recipient):
                #     return True
                # else:
                #     return False

                # （极性一样时：对象是否一样 即 是否同一条款，都兼容）

            else:
                # return False
                # （极性不一样时：若对象一样（即相同条款）则不兼容；若不同对象（即不同条款）没关系 则兼容。）
                if utils.clean_recipientWords(self.recipient) == utils.clean_recipientWords(termB.recipient):
                    return False
                else:
                    return True

        else:
            # 主线
            # 找两者各自的条件列表
            condInxsA = self.condInxs
            condInxsB = termB.condInxs
            FG = True
            for kj in termB.condInxs:

                if kj not in self.condInxs:
                    FG = False
                    break

                if not termlistA[kj].isMoreStrict(termlistB[kj], [], []):
                    FG = False
                    break

            if FG:
                # 正
                if termB.atti == term_config['attiLabel_type'][4]:
                    return False
                la1 = term_config['attiType_label'][self.atti]
                la2 = term_config['attiType_label'][termB.atti]
                la3 = term_config['atti_moreStrictTable'][la1 - 1][la2 - 1]
                if la3 == la1:
                    return True
                    # if utils.clean_recipientWords(self.recipient) == utils.clean_recipientWords(termB.recipient):
                    #     return True
                    # else:
                    #     return False
                else:
                    # return False
                    if utils.clean_recipientWords(self.recipient) == utils.clean_recipientWords(termB.recipient):
                        return False
                    else:
                        return True
            else:
                # 反
                if termB.atti == term_config['attiLabel_type'][4]:
                    return False
                la1 = term_config['turn_oppo'][term_config['attiType_label'][self.atti]-1]
                la2 = term_config['turn_oppo'][term_config['attiType_label'][termB.atti]-1]
                la3 = term_config['atti_moreStrictTable'][la1 - 1][la2 - 1]
                if la3 == la1:
                    return True
                    # if utils.clean_recipientWords(self.recipient) == utils.clean_recipientWords(termB.recipient):
                    #     return True
                    # else:
                    #     return False
                else:
                    # return False
                    if utils.clean_recipientWords(self.recipient) == utils.clean_recipientWords(termB.recipient):
                        return False
                    else:
                        return True











    def find_mostStrictAtti(self, termList, corr_cid):
        '''
        找其中最严格的那种atti（不用管self，self是其中的一个。。。）
        (若“最严格们”冲突 则atti='conflict')

        输出： 这个term with mostStrictAtti
        '''
        assert len(set([tt.content for tt in termList]))==1

        mostStrictOne = Term(content=self.content)
        attis = list(set([tt.atti for tt in termList])) #####
        atti_cids = {} # {str:int}

        moreStrictAtti = attis[0]
        if len(attis)>1:

            for at in attis[1:]:
                la1 = term_config['attiType_label'][moreStrictAtti]
                la2 = term_config['attiType_label'][at]
                moreStrictAtti = term_config['attiLabel_type'][term_config['atti_moreStrictTable'][la1 - 1][la2 - 1]]

                if moreStrictAtti == term_config['attiLabel_type'][4]:# 已经出现conflict （各取一个代表file即可）
                    # （没问题，就算是比如2+4>>4，只要记录对应来源cid即可，到时候顺着写filepath即可，正好是“对XXX文件夹……”）
                    atti_cids[term_config['attiLabel_type'][la1]] = corr_cid[[tt.atti for tt in termList].index(term_config['attiLabel_type'][la1])]
                    atti_cids[term_config['attiLabel_type'][la2]] = corr_cid[[tt.atti for tt in termList].index(term_config['attiLabel_type'][la2])]
                    break #####
                else:
                    atti_cids[moreStrictAtti] = corr_cid[[tt.atti for tt in termList].index(moreStrictAtti)]  #（取一个代表file即可） （记录atti_cids只对conflict有意义）

        # elif len(attis)==1 and moreStrictAtti == term_config['attiLabel_type'][4]:
            # atti_cids[term_config['attiLabel_type'][4]] = corr_cid
        elif len(attis) == 1:
            atti_cids[attis[0]] = corr_cid[0]



        mostStrictOne.set(atti=moreStrictAtti)

        return mostStrictOne, atti_cids









'''
term = Term()
term.set("Distribute","cannot")
print(term.get())
'''