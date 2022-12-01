# _*_coding:utf-8_*_
'''

许可证问题de具体修复方案

'''

import json
import logging
import os
import re
import pandas as pd
from itertools import product

from treelib import Tree, Node

from model.config import config as term_config
from Term import Term
from License import License
from LicenseDataset import Licensedataset
import utils


class LicenseRepair:
    def __init__(self, licenseTree=None, nid_filepath=None, hasPL=None, nid_textNeedTE=None, nid_matchedLnameList=None):

        self.licenseTree = licenseTree # 树结构（节点的索引、内容、层次、）
        self.nid_filepath = nid_filepath # dict {nid: str}
        self.nid_textNeedTE = nid_textNeedTE
        self.nid_matchedLnameList = nid_matchedLnameList

        self.hasPL = hasPL

        self.nid_license = {}  # dict {nid: LicenseObject}

        self.fixable_nid = []  # list[int]
        self.fixable_nid_all = []
        self.fixable_nid_pl = []
        self.fixable_nid_ch = []

        self.nid_termListFromChildren = {} #（保存一下这个信息）
        self.incomNid_termLists = {} # dict {部分nid: [list[TermObject], list[TermObject]] } # 下界和上界
        self.incomNid_filepathLists = {}
        # (和上面格式一致，只是对应换成 对应的term的对应极性的filepath。) # 一个atti对应的filepath可能是多个，用|来连接
        # 其实不用放 filepath from parent need。（反正exception的文本中不用涉及父节点。）
        # {nid: list[ dict{atti: str-filepaths} ]} 不用str-filepaths 只写nid即可 （list[nid]）（然后简化成了一个nid）

        self.incomAndFixable_nid = [] # list[int]




    def show_licenseTree(self):
        self.licenseTree.show()
        return



    def turn_into_licenseObjects(self, nlp, ld, ner_model_ee5, re_args, re_model, ac_model):
        '''
        填充了self.nid_license
        '''
        for nid in self.licenseTree.expand_tree(mode=Tree.DEPTH, sorting=False):
            if nid == 1:
                continue

            print('（条款提取）', nid, '/', len(self.licenseTree.nodes.keys()))

            ntag = self.licenseTree[nid].tag
            nname = self.nid_filepath[nid].split('/')[-1].replace(':','.')
            ###
            li = License(name=nname,text=ntag, textNeedTE=self.nid_textNeedTE[nid], matchedLnameList=self.nid_matchedLnameList[nid])
            li.termExtraction(nlp, ld, ner_model_ee5, re_args, re_model, ac_model)
            self.nid_license[nid] = li

        return



    def search_fixable_places(self, nlp):
        '''
        licenseTree的节点的tag，检查里面可能存在的copyright holder信息,
        （1）PL有copyright holder信息，再看内层有没有，最终可能修复至少一个位置
        （2）PL若无对应，那只能修复PL这一个位置

        :return:返回对应位置的nid，列表。
        '''
        self.fixable_nid = []
        self.fixable_nid_all = []
        self.fixable_nid_pl = []
        self.fixable_nid_ch = []
        '''先找到PL的'''
        PL_holders = []
        for nid in self.licenseTree.expand_tree(mode=Tree.DEPTH, sorting=False):

            if nid >= 2:
                self.fixable_nid_all.append(nid)


            if self.licenseTree.level(nid) == 1:

                self.fixable_nid.append(nid) #####
                self.fixable_nid_pl.append(nid)

                ntag = self.licenseTree[nid].tag
                # 检查ntag的内容
                text = ntag
                sentsList = utils.sentences_split(text)
                for sent in sentsList:
                    if utils.check_text_for_CPS(sent):  # （存在copyright相关语句）
                        print('存在CPS格式的句子：', sent)
                        holders = utils.identify_PERSON_ORGANIZATION_by_corenlp(nlp, sent)
                        if holders:
                            PL_holders.extend(holders)
                            print('存在PL_holder的句子：', holders, ' ::: ', sent)
                            self.fixable_nid_ch.append(nid)
                        #else:
                        #    PL_holder_possibleSent.append(sent)
        print("PL_holders: ", PL_holders)

        '''再看里面的'''
        for nid in self.licenseTree.expand_tree(mode=Tree.DEPTH, sorting=False):
            if self.licenseTree.level(nid) > 1:

                #CL_holder_possibleNid = False ##

                print('（检查CPS）', nid, '/', len(self.licenseTree.nodes.keys()))

                ntag = self.licenseTree[nid].tag
                # 检查ntag的内容
                text = ntag   ## .lower()
                CL_holders = [] ##
                sentsList = utils.sentences_split(text)
                for sent in sentsList:
                    if utils.check_text_for_CPS(sent):  # （存在copyright相关语句）
                        print('存在CPS格式的句子：', sent)
                        holders = utils.identify_PERSON_ORGANIZATION_by_corenlp(nlp, sent)
                        if holders:
                            CL_holders.extend(holders)
                        #elif PL_holder_possibleSent and utils.existsSameSent(PL_holder_possibleSent, sent):
                        #    CL_holder_possibleNid = True

                print('它的CL_holder：', CL_holders)

                if set(CL_holders) & set(PL_holders): ##
                    self.fixable_nid.append(nid) #####
                    self.fixable_nid_ch.append(nid)
                #elif CL_holder_possibleNid: ##
                #    print('它存在CL_holder_possibleNid.')
                #    self.fixable_nid.append(nid)

        return

    def isConflictNeed(self, termList):
        '''
        这个termlist本身内部是否存在矛盾(存在不一致)

        （这个版本是针对于 termList放的是杂七杂八 极性不同的都放一次……）
        '''
        for tt in product(termList,termList):
            if tt[0].isconflict(tt[1]):
                return True
        return False
    def isConflictNeed2(self, termList):
        '''
        （这个版本是针对于 termList放的是 某条款只放一次 只不过极性冲突的已经用'conflict'来表示了）
        '''
        for tt in termList:
            if tt.isconflict2():
                return True
        return False

    def getConflictNeed2(self, termList):
        conf_tt_j_list = []
        for j, tt in enumerate(termList):
            if tt.isconflict2():
                conf_tt_j_list.append(j)
        return conf_tt_j_list


    def getConflictNeeds(self, termList):
        '''
        这个termlist本身内部 存在的矛盾 的具体位置情况
        '''
        cfTupIndList0 = [] # 先每个元组是一对id
        for i in range(len(termList)):
            for j in range(i+1, len(termList)):
                if termList[i].isconflict(termList[j]):
                    cfTupIndList0.append((i,j))
        # 保证关于同term.content的只出现一个元组 （每个元组是》=2个id）
        cfTupIndList = []
        for i in range(len(cfTupIndList0)):
            tp1 = cfTupIndList0[i]
            tmp = list(tp1)
            for j in range(i+1, len(cfTupIndList0)):
                tp2 = cfTupIndList0[j]
                if tp1 != tp2 and termList[tp1[0]].isSameContent(termList[tp2[0]]):
                    tmp.extend(list(tp2))
            tmp = set(tmp)
            fg = False
            for tp in cfTupIndList:
                if set(tp).issuperset(tmp):
                    fg = True
            if not fg:
                cfTupIndList.append(tuple(list(tmp)))
        cfTupIndList = list(set(cfTupIndList))
        return cfTupIndList


    def repair_choose_popular_licenses(self, termlist_need_fromChildren, termlist_need_fromParent, ld):
        '''
        判断本数据库中 是否存在满足此需求的许可证 【（满足这个范围要求）】
        OK。
        '''
        '''
        ！！！！奥 确实可以顺便推荐改动最小的方案，，，，
        '''
        return ld.isNeedSatisfied_2(termlist_need_fromChildren, termlist_need_fromParent)


    def repair_generate_one_custom_license(self,termlist_need_fromChildren, termlist_need_fromParent):
        '''
        '''
        l_custom = ''
        termContent_template = utils.read_custom_template()
        for tt in termlist_need_fromChildren:
            template = termContent_template[tt.content]
            l_custom += ('You '+tt.atti+' '+ template + '.'+'\n')
        return l_custom

    def repair_generate_one_custom_license_2(self, termlist_need_fromChildren, termlist_need_fromParent, nid, nlp, ner_model_ee5, re_args, re_model):
        text = ''

        for j in range(23):
            # 每个条款下 可能添加至少一句话

            # （有exception的是有多个atti；没有的是只有一个atti；那反正遍历atti就行了呗；总而言之 每次给到一句话）
            atti_cid = self.incomNid_filepathLists[nid][j]  # dict{ atti-str: nid-int}
            for atti, cid in atti_cid.items():
                cidFilepathList = self.nid_filepath[cid]
                # 添加1句话
                # （去找cid结点里j条款的说辞（并解构））
                ll = self.nid_license[cid]
                # 得到对应的tr对象
                # (组装成一句话)
                # tr = ll.extract_termRelated(nlp, ner_model_ee5, re_args, re_model, j)
                # sent = tr.composeOneSent()
                sent = ll.termList[j].composeOneSent(ll.termList)

                text += sent

            text += '\n'

        return text




    def repair_generate_one_exception_license(self, termList, termList_filepathList, cfTupIndList, ):
        '''

        :param termList:
        :param termList_filepathList:
        :param cfTUupIndList:
        :return:
        '''
        l_exception = ''
        termContent_template = utils.read_custom_template()
        termList_alre = [False]*len(termList)

        for tp in cfTupIndList:
            template = termContent_template[termList[tp[0]].content]
            for k in tp:
                k_atti = termList[k].atti
                k_obj = termList_filepathList[k]
                l_exception += ('For the code in ' + k_obj + ', you ' + k_atti + ' ' + template + '; ')
                termList_alre[k] = True

            l_exception += '\n'

        l_exception += 'The other terms are below: ' + '\n'

        for i in range(len(termList)):
            if not termList_alre[i]:
                tt = termList[i]
                template = termContent_template[tt.content]
                l_exception += ('You ' + tt.atti + ' ' + template + '.' + '\n')

        return l_exception

    def repair_generate_one_exception_license_2(self, termList, nid, conf_tt_j_list, nlp, ner_model_ee5, re_args, re_model):
        '''

        （等兼容性检测那里填充好self.incomNid_filepathLists，这里就按那个数据结构来写）

        :param termList:
        :param termList_filepathList:
        :return:
        '''

        # return '(we will generate a exception license for you ...)'

        '''
        
        '''
        text = ''

        for j in range(23):
            # 每个条款下 可能添加至少一句话

            if j in conf_tt_j_list:
                # 当前条款的极性有exception时
                text += 'ONE EXCEPTION: '

            # （有exception的是有多个atti；没有的是只有一个atti；那反正遍历atti就行了呗；总而言之 每次给到一句话）
            atti_cid = self.incomNid_filepathLists[nid][j]  # dict{ atti-str: nid-int}
            for atti, cid in atti_cid.items():

                cidFilepath = self.nid_filepath[cid]
                if j in conf_tt_j_list:
                    text += 'For the code in : '+cidFilepath+', '

                # 添加1句话
                # （去找cid结点里j条款的说辞（并解构））
                ll = self.nid_license[cid]
                # tr = ll.extract_termRelated(nlp, ner_model_ee5, re_args, re_model, j)
                # # 得到对应的tr对象
                # # (组装成一句话)
                # sent = tr.composeOneSent()
                sent = ll.termList[j].composeOneSent(ll.termList)

                text += sent

            text += '\n'

        return text




    def repair_onePlace(self, nid, ld, nlp, ner_model_ee5, re_args, re_model):
        '''
        输入：本次待修复的位置nid
        输出：给此位置的修复建议（一段文本）
        '''
        '''
        # （一些测试参数）
        termList = [
            Term('Distribute', 'can'),
            Term('Modify', 'can'),
            Term('Commercial Use', 'cannot'),
            Term('Hold Liable', 'cannot'),
            Term('Include Copyright', 'must'),
            Term('Sublicense', 'can'),
            Term('Disclose Source', 'must'),
            Term('Rename', 'must'),
        ]

        termList_filepathList = []
        '''

        # 该位置的已知信息
        termlist_need_fromChildren = self.incomNid_termLists[nid][0]
        termlist_need_fromParent = self.incomNid_termLists[nid][1]
        # termlist_real = self.nid_license[nid].termList
        # termList_filepathList = [] #self.incomNid_filepathLists[nid]

        print('【【【【termlist_need_fromChildren: ', ' '.join([str(term_config['attiType_label'][tt.getAtti()]) for tt in termlist_need_fromChildren]))


        # 修复过程

        if self.isConflictNeed2(termList=termlist_need_fromChildren):
            print("【需求存在矛盾，生成带有exception的自定义许可证】")
            # cfTupIndList = lr.getConflictNeeds(termList=termList)
            conf_tt_j_list = self.getConflictNeed2(termList=termlist_need_fromChildren)
            text = self.repair_generate_one_exception_license_2(termList=termlist_need_fromChildren, nid=nid, conf_tt_j_list=conf_tt_j_list,
                                                                nlp=nlp, ner_model_ee5=ner_model_ee5, re_args=re_args, re_model=re_model)
            return 1, text


        else:
            # termlist_need_fromParent肯定不含有‘conflict’
            # termlist_need_fromChildren若有的话会进上面的exception，因此下面popular和custom肯定是有效的atti。


            abledList = self.repair_choose_popular_licenses(termlist_need_fromChildren, termlist_need_fromParent, ld)
            if not abledList:
                print("【数据库无法满足需求，生成自定义许可证】")
                text = self.repair_generate_one_custom_license_2(termlist_need_fromChildren=termlist_need_fromChildren,
                                                                 termlist_need_fromParent=termlist_need_fromParent,
                                                                 nid=nid,  nlp=nlp, ner_model_ee5=ner_model_ee5, re_args=re_args, re_model=re_model)
                return 3, text

            else:
                print("【数据库满足需求，推荐以下已有许可证】")
                return 2, str([ll.name for ll in abledList])





    def isCompatible_real_for_needs(self, nid, needtermlist):
        '''
        比较两个termlist（一个节点上的，本身VS被需求）

        输入：两个termlist
        输出：是否。

        》》每个term.content上 本身atti 应该比 被需求atti 一样or更加严格。
        '''

        realTermlist = self.nid_license[nid].termList
        # print(nid, realTermlist, needtermlist)
        if not realTermlist or not needtermlist:
            print(nid, realTermlist, self.nid_license[nid].matchedLnameList)

        if not realTermlist:
            return True

        '''
        （暂时简化成按顺序直接就term.content已经对应了）
        '''
        #print(nid, [tt.atti for tt in realTermlist], [tt.atti for tt in needtermlist])

        for j in range(23):

            if not realTermlist[j].isMoreStrict(needtermlist[j], realTermlist, needtermlist):
                #print(j, realTermlist[j].atti, needtermlist[j].atti)
                return False

        return True


    def get_oneNode_needs_from_its_childern(self, termlists_of_cid):
        '''
        得到此节点的低层需求termlist，从其所有子节点的termlist。
        输入：若干个termlist
        输出：一个termlist。

        》》每个term.content上 找其中最严格的那种atti。
        若“最严格们”冲突 则atti='conflict'（下游直接就不兼容了）
        '''
        termlist = []
        attiCidsList = []

        for j in range(23):

            terms_sameCont_diffAtti = []
            corr_cid = []
            for cid in termlists_of_cid.keys():
                termlist_tmp = termlists_of_cid[cid]
                if not termlist_tmp:
                    continue
                tt = termlist_tmp[j]
                # 设置缺省认定值
                # tt.set_absentAtti()
                terms_sameCont_diffAtti.append(tt)
                corr_cid.append(cid)
            # 找其中最严格的那种atti
            mostStrictOne, atti_cids = terms_sameCont_diffAtti[0].find_mostStrictAtti(terms_sameCont_diffAtti, corr_cid)
            termlist.append(mostStrictOne)
            attiCidsList.append(atti_cids)

        return termlist, attiCidsList


    def upward_get_allNodes_needs_from_childern(self):
        '''
        逐层向上，对于非叶子结点，得到各自的低层需求termlist。

        找非叶子节点，
        按深度排序，
        （保证在计算它时，它的所有子节点已经计算过）
        （遍历其所有子节点的termlist：其中若为叶子则使用其本身termlist/若为非叶子则用其需求termlist。）

        按深度排序then依次计算。《《《《 先这样写。
        or
        写一个递归函数
        '''
        nid_termListFromChildren = {}
        nid_attiCidsListFromChildren = {}

        nids_of_leaves = [nd.identifier for nd in self.licenseTree.leaves()]
        nids_of_not_leaves = set(list(self.licenseTree.nodes.keys())) - set(list([1])) - set(nids_of_leaves) ###
        nid_level = dict(zip(nids_of_not_leaves, [self.licenseTree.level(nid) for nid in nids_of_not_leaves]))
        sorted_nid_level = sorted(nid_level.items(), key=lambda d:d[1], reverse=True)

        for nid, nlevel in sorted_nid_level:
            # 找到所有子节点
            childrenList = self.licenseTree.is_branch(nid)

            termlists_of_cid = {}
            # 找到子节点的termlist（若为叶子则使用其本身termlist/若为非叶子则用其需求termlist）
            for cid in childrenList:
                # 每一个子节点：
                assert cid in nids_of_leaves or cid in nid_termListFromChildren.keys()
                '''
                if cid in nids_of_leaves:
                    termlists_of_cid[cid] = self.nid_license[cid].termList
                else:
                    termlists_of_cid[cid] = nid_termListFromChildren[cid]
                '''
                termlists_of_cid[cid] = self.nid_license[cid].termList #### 【1109确定的版本】

                ############################
                # if nid in [4,48,51]:


                ###########################


            # 更新nid_termListFromChildren
            termlist_from_children, attiCidsList_from_children = self.get_oneNode_needs_from_its_childern(termlists_of_cid)
            nid_termListFromChildren[nid] = termlist_from_children
            nid_attiCidsListFromChildren[nid] = attiCidsList_from_children

        '''
        （但为了get_PL_needs_from_childern万一从叶子，》》nid_termListFromChildren也放入叶子的本身。）
        '''
        for nid  in nids_of_leaves:
            nid_termListFromChildren[nid] = self.nid_license[nid].termList


        return nid_termListFromChildren, nid_attiCidsListFromChildren


    def get_PL_needs_from_childern(self):
        '''
        在项目不含PL时(self.hasPL=False):
            填充 self.incomNid_termList[-1] 和 incomNid_filepathLists[-1]
        （此时已经计算完了全OSS的层次化兼容性检测，在此基础上，找第一层 for PL）
        '''
        termlists_of_cid = {}
        for nid in self.nid_termListFromChildren.keys():
            if self.licenseTree.level(nid) == 1:
                termlists_of_cid[nid] = self.nid_termListFromChildren[nid]

        termlist_from_children, attiCidsList_from_children = self.get_oneNode_needs_from_its_childern(termlists_of_cid)
        self.incomNid_termLists[-1] = [termlist_from_children, []]
        self.incomNid_filepathLists[-1] = attiCidsList_from_children

        return




    def get_incomNodes_needs_from_parent(self, nid):
        '''
        对那些不兼容的位置，只向上看一层，，
        '''
        nParid = self.licenseTree.parent(nid).identifier
        termlist_from_parent = self.nid_license[nParid].termList

        return termlist_from_parent


    def detect_incompatibility_hierarchically(self):
        '''
        从最内层向外 汇总当前位置被内层导致的需求 判断当前位置是否发生了不兼容
        （以一个项目即一个子树为单位）

        使用：self.licenseTree，self.nid_license；self.nid_filepath。

        最终结果：【填充self.incomNid_termList】和incomNid_filepathLists。

        1. 逐层向上，得到各自的低层需求termlist。（非叶子结点）（但为了get_PL_needs_from_childern万一从叶子，》》nid_termListFromChildren也放入叶子的本身。）
            【1109改成了 “不传递 只看当前的父子关系”】
        2. 比较各自的需求termlist和本身termlist，得到发生不兼容的点。（不兼容and非叶子节点）
        3. 逐层向下只向上看一层，对那些不兼容的位置 根据其高层需求得到各自的高层需求termlist。（“只为了修复时不至于产生新的冲突”）  （不兼容and非叶子结点and非根节点）

        '''
        # 1
        self.nid_termListFromChildren, nid_attiCidsListFromChildren = self.upward_get_allNodes_needs_from_childern()

        # 2
        for nid, needtermlist in self.nid_termListFromChildren.items():
            if not self.isCompatible_real_for_needs(nid, needtermlist):
                self.incomNid_termLists[nid] = [needtermlist] # 添加下界
                self.incomNid_filepathLists[nid] = nid_attiCidsListFromChildren[nid]
        print(self.incomNid_termLists.keys())
        print(self.incomNid_filepathLists)

        # 3
        for icNid in self.incomNid_termLists.keys():
            if self.licenseTree.level(icNid) > 1:
                termlist_from_parent = self.get_incomNodes_needs_from_parent(icNid)
                self.incomNid_termLists[icNid].append(termlist_from_parent) # 添加上界
            else:
                self.incomNid_termLists[icNid].append([])


        return




    def get_incom_and_fixable_places(self):
        '''
        填充self.incomAndFixable_nid，列表
        '''
        incom_nids = self.incomNid_termLists.keys()
        fixable_nids = self.fixable_nid

        self.incomAndFixable_nid = list(set(incom_nids) & set(fixable_nids))
        return


    def getShortPath(self,nid,repoName):
        rootDir = os.path.dirname(os.path.abspath(__file__))
        return self.nid_filepath[nid][len(os.path.join(os.path.dirname(rootDir), 'repos', repoName)+'/'):]


    def baseline_tool_nonhiera(self, repoName):

        reportList = []
        if self.hasPL:
            termlist_PL = self.nid_license[2].termList
            for nid in self.licenseTree.expand_tree(mode=Tree.DEPTH, sorting=False):
                if nid <= 2:
                    continue
                termlist_CLi = self.nid_license[nid].termList
                ##
                tmp_j_list = []
                for j in range(23):
                    if not termlist_PL[j].isMoreStrict(termlist_CLi[j], termlist_PL, termlist_CLi):  ##
                        # tmp_j_list.append(j)
                        tmp_j_list.append(term_config['term_list'][j])
                if tmp_j_list:
                    sent = {}
                    sent['A'] = self.getShortPath(nid=2, repoName=repoName)
                    sent['B'] = self.getShortPath(nid=nid, repoName=repoName)
                    sent['incomterms'] = ', '.join(tmp_j_list)
                    reportList.append(sent)
        else:

            PL = License(name='PL')
            for j in range(23):
                content = term_config['term_list'][j]
                tt = Term(content=content)
                tt.set_all_default()
                PL.addTerm(tt)  ###

            termlist_PL = PL.termList
            for nid in self.licenseTree.expand_tree(mode=Tree.DEPTH, sorting=False):
                if nid <= 1:
                    continue
                termlist_CLi = self.nid_license[nid].termList
                ##
                tmp_j_list = []
                for j in range(23):
                    if not termlist_PL[j].isMoreStrict(termlist_CLi[j], termlist_PL, termlist_CLi):  ##
                        # tmp_j_list.append(j)
                        tmp_j_list.append(term_config['term_list'][j])
                if tmp_j_list:
                    sent = {}
                    sent['A'] = self.getShortPath(nid=2, repoName=repoName)
                    sent['B'] = self.getShortPath(nid=nid, repoName=repoName)
                    sent['incomterms'] = ', '.join(tmp_j_list)
                    reportList.append(sent)

            # print()
            # cids = []
            # for nid in self.licenseTree.expand_tree(mode=Tree.DEPTH, sorting=False):
            #     if nid == 1:
            #         continue
            #     cids.append(nid)
            # for d1 in range(0, len(cids)):
            #     for d2 in range(d1 + 1, len(cids)):
            #         termlist_CL1 = self.nid_license[cids[d1]].termList
            #         termlist_CL2 = self.nid_license[cids[d2]].termList
            #         ##
            #         tmp_j_list = []
            #         for j in range(23):
            #             if not termlist_CL1[j].isTwoOccurConflict(termlist_CL2[j]):  ##
            #                 # tmp_j_list.append(j)
            #                 tmp_j_list.append(term_config['term_list'][j])
            #         if tmp_j_list:
            #             sent = {}
            #             sent['A'] = self.getShortPath(nid=cids[d1], repoName=repoName)
            #             sent['B'] = self.getShortPath(nid=cids[d2], repoName=repoName)
            #             sent['incomterms'] = ', '.join(tmp_j_list)
            #             reportList.append(sent)

        return reportList









'''模块案例测试'''
def runLicenseRepair(repo, nlp, ld, ner_model_ee5, re_args, re_model, ac_model):
    '''
    输入：项目名 （默认其在文件夹./unzips/内）
    输出：修复结果，以及lr的一些属性统计数据，
    调试信息会适当地控制台输出
    '''
    print('repo: ', repo)


    # 生成许可证树
    # import projectLicenseTree
    from projectLicenseTree import get_license_tree
    print('开始构建许可证树……')
    licenseTree, nid_filepath, hasPL, nid_textNeedTE, nid_matchedLnameList = get_license_tree(repo=repo)  # nid_filepath 每个叶子结点所对应的文件路径。
    print('hasPL: ', hasPL)
    for key, val in nid_matchedLnameList.items():
        print(key, val)


    lr = LicenseRepair(licenseTree=licenseTree, nid_filepath=nid_filepath, hasPL=hasPL,
                       nid_textNeedTE=nid_textNeedTE, nid_matchedLnameList=nid_matchedLnameList)
    # lr.show_licenseTree()

    # 遍历输出看一下
    print('关于projectLicenseTree的一些遍历信息：')
    for nid in lr.licenseTree.expand_tree(mode=Tree.DEPTH, sorting=False):
        if nid == 1:
            continue
        # （试用一些个函数）
        ntag = lr.licenseTree[nid].tag
        nidd = lr.licenseTree[nid].identifier
        npath = lr.nid_filepath[nid]
        nlevel = lr.licenseTree.level(nid)  # PL的level=1.
        nparent = lr.licenseTree.parent(nid).identifier
        nchildren = lr.licenseTree.is_branch(nid)
        # print('\t'.join([str(key),val[len('D:\Python\OSSL2//unzips/'):]]))
        print('\t'.join([str(nid), str(nidd), str(nlevel), npath, str(nparent), str(nchildren)]))
    print('所有结点：', lr.licenseTree.nodes.keys())
    print('叶子结点：', [nd.identifier for nd in lr.licenseTree.leaves()])

    if len(lr.licenseTree.leaves())==1 and lr.licenseTree.leaves()[0].identifier==1:
        return lr, lr.hasPL, 0, 0, 0, []


    # 找有权限的位置
    if lr.hasPL:
        lr.search_fixable_places(nlp=nlp)
    else:
        lr.fixable_nid.append(-1)
        lr.fixable_nid_pl.append(-1)
        lr.fixable_nid_all.append(-1)
    print('找到可修复的位置：')
    print('lr.fixable_nid: ', len(lr.fixable_nid), lr.fixable_nid)


    # 每个许可证节点，生成对应的license对象
    # 条款提取 （填充self.nid_license）
    print('开始进行条款提取 都对应生成License对象……')
    lr.turn_into_licenseObjects(nlp, ld, ner_model_ee5, re_args, re_model, ac_model)



    # 层次兼容性检测
    # （找到发生不兼容的位置 及其需求）（填充self.incomNid_termList）
    print('开始进行层次化的兼容性检测……')
    lr.detect_incompatibility_hierarchically()

    if not lr.hasPL:  # 需要计算得到'nid=-1'时的self.incomNid_termLists
        lr.get_PL_needs_from_childern()


    # 找到不兼容and能修复的位置
    if lr.hasPL:
        lr.get_incom_and_fixable_places()
    else:  # (此时不管是否兼容 反正都得生成一个新的PL。)
        lr.incomAndFixable_nid.append(-1)
    print('找到发生不兼容且我们能修复的位置：')
    print('lr.incomAndFixable_nid: ', len(lr.incomAndFixable_nid), lr.incomAndFixable_nid)


    # 修复
    print('开始修复……')
    repairMethod = []

    DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
    fw = open(os.path.join(DIR, 'REPAIRED', repo + '.json'), 'w', encoding="utf-8")
    REPAIRED_DATA = []


    for nid_to_repair in lr.incomAndFixable_nid:
        print('====================================================================')
        print('将要修复的位置：', nid_to_repair)
        print('所在文件路径：', lr.nid_filepath[nid_to_repair])
        ## 修复
        repairMethod_i, licenseText_repaired = lr.repair_onePlace(nid=nid_to_repair, ld=ld,
                                                                  nlp=nlp, ner_model_ee5=ner_model_ee5,
                                                                  re_args=re_args, re_model=re_model)
        repairMethod.append(repairMethod_i)
        print('修复完成。')
        print('建议该位置的许可证文本改为如下:', licenseText_repaired)

        REPAIRED_DATA.append({'nid':nid_to_repair,
                              'filepath':lr.nid_filepath[nid_to_repair],
                              'method':repairMethod_i,
                              'text': licenseText_repaired})

    json.dump(REPAIRED_DATA, fw)
    fw.close()




    return lr, lr.hasPL, len(lr.fixable_nid), len(lr.incomNid_termLists), len(lr.incomAndFixable_nid), repairMethod



