# _*_coding:utf-8_*_
'''
'''

import json
import logging
import os
import re
import sys

import pandas as pd
import numpy as np
import pickle
import h5py
from tqdm import tqdm, trange
from collections import defaultdict

import torch
# from transformers import AutoTokenizer, RobertaTokenizer, BertTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from model.config import config as term_config


'''
文本处理相关
'''

def cleanText(text):
    '''
    对英文做简单的数据清洗预处理
    【全部设为小写。】

    用它处理过的：
    textTagging的准备数据，


    '''
    punctuation = """.,?!:;(){}[]<>@$#%&"""
    text = text.lower().replace('\n', ' ')
    text = text.lower().replace('\t', ' ')
    text = text.lower().replace(r'\r', ' ')
    text = text.lower().replace(r'\t', ' ')
    text = text.lower().replace(r'\n', ' ')
    text = text.lower().replace(r'\x', ' ')
    text = text.lower().replace('\\', ' ')
    text = text.lower().replace('/', ' ')

    text = text.replace('<br />', ' ')
    for c in punctuation:
        text = text.replace(c, ' %s ' % c)
    # corpus = [removeStopwords(z) for z in corpus]
    text = text.replace('  ', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('  ', ' ')
    return text


def clean_recipientWords(text):
    text = text.strip().lower()
    text = text.replace('  ', ' ')

    return text



def sentences_split(text):
    '''
    先简单写一下，
    之后用corenlp做
    :param text:
    :return:
    '''
    return text.split('.')





'''
文本检查和搜索相关
'''
def read_text(filepath):
    text = ''
    try:
        with open(filepath, 'r', encoding="utf-8")as fr:
            for line in fr.readlines():
                line = line.strip()
                if line:
                    text += line + ' '
        fr.close()
    except Exception as e:
        print(e)
        print('from【read_text】')
        print(filepath)

    return text

def checkLicenseFileName(filename):
    if re.findall(r'^license$', filename, flags=re.IGNORECASE) or re.findall(r'^license\.[a-zA-Z]+', filename, flags=re.IGNORECASE) \
            or re.findall(r'^copying$', filename, flags=re.IGNORECASE) or re.findall(r'^copying\.[a-zA-Z]+', filename, flags=re.IGNORECASE) \
            or re.findall(r'^readme.md$', filename, flags=re.IGNORECASE) \
            or re.findall(r'^setup.py$', filename, flags=re.IGNORECASE) \
            or re.findall(r'^__pkginfo__.py$', filename, flags=re.IGNORECASE):
        return True
    else:
        return False

def extract_comments_in_pyFile(filepath):
    targetText = ""
    try:
        with open(filepath, 'r', encoding="utf-8") as fr:
            fg = False
            for line in fr.readlines():
                if line.strip().startswith("#"):
                    targetText += line.strip()[1:].strip() + ' '
                elif line.strip().startswith("\'\'\'") or line.strip().startswith("\"\"\""):
                    if not fg:
                        # start ...
                        if line.strip().endswith("\'\'\'", 3, len(line.strip())) or line.strip().endswith("\"\"\"", 3,
                                                                                                          len(
                                                                                                                  line.strip())):
                            targetText += line.strip()[3:-3].strip() + ' '
                        else:
                            targetText += line.strip()[3:].strip() + ' '
                            fg = True
                    else:
                        fg = False
                elif line.strip():
                    if fg:
                        targetText += line.strip() + ' '
                    else:
                        break
        fr.close()
    except Exception as e:
        print(e)
        print('from【extract_comments_in_pyFile】')
        print(filepath)

    return targetText

# targetText = extract_comments_in_pyFile('D:\Python\_LicenseAnalysis_2\scrapy/scrapy_list.py')
# print(targetText)


def get_licenseNameList1(license_check_filepath):
    license_check = {}
    license_name = {}
    with open(license_check_filepath, 'r', encoding='utf-8') as f:
        for li in f.readlines():
            li = li.strip()
            if li:
                name = ' '.join(li.split(' ')[2:])
                name_list = []
                if re.findall("\(", name):
                    name_list.append(name.split(' ')[-1][1:-1])
                    name_list.append(' '.join(name.split(' ')[:-1]))
                else:
                    name_list.append(name)
                license_name[li.split(' ')[0]] = ' '.join(li.split(' ')[2:])
                license_check[li.split(' ')[0]] = name_list
    f.close()
    return license_check, license_name

def get_licenseNameList2(dirpath):
    licenseNameList = []
    for file in os.listdir(dirpath):
        licenseNameList.append(file[:-4])
    return licenseNameList

def get_licenseTextDict2(dirpath):
    licenseTextDict = {}
    for file in os.listdir(dirpath):
        text = read_text(os.path.join(dirpath, file))
        licenseTextDict[file[:-4]] = text
    return licenseTextDict


def check_text_licenseNameList1(license_check,text):
    for i in range(212):
        filter_list = license_check[str(i + 1)]
        for kk in filter_list:
            if text.upper().find(kk.upper()) > -1:
                return True
    return False

def check_text_licenseNameList2(licenseNameList,text):
    matchedLnameList = []

    for i in range(len(licenseNameList)):
        liname = licenseNameList[i]
        n1 = liname
        n2 = ' '.join(liname.split('-'))
        if text.upper().find(n1.upper()) > -1:
            matchedLnameList.append(liname)
        elif text.upper().find(n2.upper()) > -1:
            matchedLnameList.append(liname)

    ## 再把一些常见的单独去找吧（即使OSS可能表述不完全规范，，，）
    if text.upper().find('(BSD)'.upper()) > -1 \
            or text.upper().find('(BSD-3)'.upper()) > -1 \
            or text.upper().find('BSD License'.upper()) > -1 \
            or text.upper().find('New BSD'.upper()) > -1 \
            or text.upper().strip() == 'BSD':
        matchedLnameList.append('BSD-3-Clause')
    if text.upper().find('(PSF)'.upper()) > -1 or text.upper().find('Python Software Foundation License'.upper()) > -1:
        matchedLnameList.append('PSF-2.0')
    if text.upper().find('License: MIT'.upper()) > -1 or text.upper().find('License: Public Domain'.upper()) > -1:
        matchedLnameList.append('MIT License')
    if text.upper().find('Apache License, Version 2.0'.upper()) > -1 or text.upper().find('Apache Software License'.upper()) > -1:
        matchedLnameList.append('Apache-2.0')
    if text.upper().find('license: GPLv3'.upper()) > -1 or text.upper().find('GPL-3.0'.upper()) > -1 or text.upper().find(
            'GPLv3'.upper()) > -1 \
            or text.upper().find('GNU General Public License Version 3'.upper()) > -1:
        matchedLnameList.append('GPL-3.0-only')
    if text.upper().find('license: GPLv2'.upper()) > -1 or text.upper().find('GPL-2.0'.upper()) > -1 or text.upper().find(
            'GPLv2'.upper()) > -1 \
            or text.upper().find('GNU General Public License Version 2'.upper()) > -1:
        matchedLnameList.append('GPL-2.0-only')
    if text.upper().find('(LGPL)'.upper()) > -1 or text.upper().find(
            'GNU Library or Lesser General Public License'.upper()) > -1:
        matchedLnameList.append('LGPL-2.1-only')
    if text.upper().find('(AGPL)'.upper()) > -1 or text.upper().find('GNU AFFERO GENERAL PUBLIC LICENSE'.upper()) > -1:
        matchedLnameList.append('LGPL-2.1-only')

    print('     找到的ref-license: ', matchedLnameList)
    if matchedLnameList:
        return True
    return False


def check_text_licenseNameList2_2(licenseNameList,ll, dirpath):
    matchedLnameList = []

    for i in range(len(licenseNameList)):
        liname = licenseNameList[i]
        n1 = liname
        n2 = ' '.join(liname.split('-'))
        if ll.upper().find(n1.upper()) > -1:
            matchedLnameList.append(liname)
        elif ll.upper().find(n2.upper()) > -1:
            matchedLnameList.append(liname)

    ## 再把一些常见的单独去找吧（即使OSS可能表述不完全规范，，，）
    if ll.upper().find('(BSD)'.upper()) > -1 \
            or ll.upper().find('(BSD-3)'.upper()) > -1 \
            or ll.upper().find('BSD License'.upper()) > -1 \
            or ll.upper().find('New BSD'.upper()) > -1 \
            or ll.upper().strip() == 'BSD':
        matchedLnameList.append('BSD-3-Clause')
    if ll.upper().find('(PSF)'.upper()) > -1 or ll.upper().find('Python Software Foundation License'.upper()) > -1:
        matchedLnameList.append('PSF-2.0')
    if ll.upper().find('License: MIT'.upper()) > -1 or ll.upper().find('License: Public Domain'.upper()) > -1:
        matchedLnameList.append('MIT License')
    if ll.upper().find('Apache License, Version 2.0'.upper()) > -1 or ll.upper().find('Apache Software License'.upper()) > -1:
        matchedLnameList.append('Apache-2.0')
    if ll.upper().find('license: GPLv3'.upper()) > -1 or ll.upper().find('GPL-3.0'.upper()) > -1 or ll.upper().find('GPLv3'.upper()) > -1\
            or ll.upper().find('GNU General Public License Version 3'.upper()) > -1:
        matchedLnameList.append('GPL-3.0-only')
    if ll.upper().find('license: GPLv2'.upper()) > -1 or ll.upper().find('GPL-2.0'.upper()) > -1 or ll.upper().find('GPLv2'.upper()) > -1\
            or ll.upper().find('GNU General Public License Version 2'.upper()) > -1:
        matchedLnameList.append('GPL-2.0-only')
    if ll.upper().find('(LGPL)'.upper()) > -1 or ll.upper().find('GNU Library or Lesser General Public License'.upper()) > -1:
        matchedLnameList.append('LGPL-2.1-only')
    if ll.upper().find('(AGPL)'.upper()) > -1 or ll.upper().find('GNU AFFERO GENERAL PUBLIC LICENSE'.upper()) > -1:
        matchedLnameList.append('LGPL-2.1-only')
    if ll.upper().find('Creative Commons Attribution-Share Alike'.upper()) > -1 :
        matchedLnameList.append('CC-BY-SA-3.0')

    matchedLnameList = list(set(matchedLnameList))
    print('     找到的ref-license: ', matchedLnameList)
    text = ''
    for liname in matchedLnameList:
        text += read_text(os.path.join(dirpath, liname + '.txt'))

    return text, matchedLnameList


def add_possible_refLicenseTexts(licenseNameList, text, dirpath):
    return check_text_licenseNameList2_2(licenseNameList, text, dirpath)


def get_text_except_cpsSents(text):
    '''
    清洗文本 and 去掉CPS句子
    '''
    text = cleanText(text)
    sentsList = sentences_split(text)
    sents = []
    for sent in sentsList:
        if not check_text_for_CPS(sent):  # 存在copyright相关语句
            sents.append(sent)
    return '.'.join(sents)


def match_availableText_for_possible_refLicenseTexts(text, licenseTextDict):
    #print('get_text_except_cpsSents(text): ', get_text_except_cpsSents(text))
    #print(get_text_except_cpsSents(licenseTextDict['Apache-2.0']))
    #print(licenseTextDict.keys())
    matchedLnameList = []
    for liname, litext in licenseTextDict.items():
        #print(liname)
        if get_text_except_cpsSents(text) == get_text_except_cpsSents(litext):
            matchedLnameList.append(liname)
            print('match_availableText_for_possible_refLicenseTexts: ', matchedLnameList)
            return matchedLnameList
    return matchedLnameList

def check_text_for_licenseWords(text, license_check, licenseNameList):
    if check_text_licenseNameList2(licenseNameList,text) \
            or text.lower().find('copyright') > -1 \
            or (text.lower().find('license') > -1 and
                (text.lower().find('publish') > -1 or text.lower().find('distribute') > -1
                 or text.lower().find('warranty') > -1 or text.lower().find('permission') > -1)):
        return True

    return False

def check_text_for_CPS(sent):
    '''
    判断是否存在相关字段
    :param text:
    :return:
    '''
    ks = [
        'authored by',
        'author:',
        'author=',
        'copyright:',
        'copyright (c)',
        'copyright ©',
        'copyrighted by',
        'copyright [',
        'copyright {',
        'copyright 20', # 比如：Copyright 2021 Google LLC
        'copyright (c) 20',  # 比如：Copyright (C) 2018-2022 Rocky Bernstein <rocky@gnu.org>
    ]
    for k in ks:
        if sent.lower().find(k) > -1:
            return True
    return False


def existsSameSent(sentList, ss):
    for sent in sentList:
        if ss.find(sent)>-1 or sent.find(ss)>-1:
            return True
    return False




'''
bert相关，embedding相关，
'''


def generate_bert_ids_for_sentence(tokenizer, sentence, fg):
    '''

    :param sentence:
    :return:
    '''
    ids = tokenizer.encode_plus(sentence, max_length=512, padding='max_length', truncation=True)
    if fg==1:
        ids = list(ids['input_ids'])[1:len(sentence) + 1]
    if fg==2:
        ids = list(ids['input_ids'])[1:]
    # print(ids)
    return ids



def get_str_from_bert_ids(tokenizer, ids):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    sentence = tokenizer.convert_tokens_to_string(tokens)
    return tokens

'''
# 测试一下
seq = ['hello', 'you', ',', 'i', 'am', 'Tom']
print(seq)
ids = generate_bert_ids_for_sentence(seq)
print(ids)
print(get_str_from_bert_ids(ids))
'''



def get_unique_lists_in_list(lis, isInt = False, isOneMore = False):
    '''
    list里面放list，不能直接用list(set())来去重，
    所以写了这个函数
    '''
    lis_str = [' '.join([str(id) for id in ids]) for ids in lis]
    uni_lis_str = list(set(lis_str))

    if isOneMore:
        uni_lis = [[ids_str.split(' ')[0], ' '.join(ids_str.split(' ')[1:])] for ids_str in uni_lis_str]
    else:
        uni_lis = [ids_str.split(' ') for ids_str in uni_lis_str]

    if isInt:
        uni_lis = [[int(id) for id in ids] for ids in uni_lis]

    return uni_lis



def is_subsequence(s, t):
    '''
    某个列表a是否是另一个列表b的子列表,存在一次即可,
    '''
    n = len(s)
    m = len(t)

    j = k = 0
    while j < n and k < m:
        print(k, j)
        if j>0 and s[j] != t[k]:
            j = 0
            continue
        if s[j] == t[k]:
            j += 1
        k += 1

    if j == n:
        return True
    else:
        return False

# print(is_subsequence([1,2,3,0],[1,2,3,4,5,0,1,2,3,0]))


def read_embedid_embedding(embedDir):
    '''
    读取.npy
    '''
    data = np.load(embedDir)
    print(data.shape) # (1737, 768)
    #print(data[0])
    #print(data[1])
    return data


def get_embedid_str(menStrDir, pageid2embedidDir):
    '''
    pageid2embedid，pageid是mention_str_dataset.txt的行索引。
    得到embedid2str
    '''
    embedid2str = {}
    ##
    pageid2embedid = pickle.load(open(pageid2embedidDir, 'rb'))
    print('pageid2embedid: ', type(pageid2embedid), len(pageid2embedid)) # pageid2embedid:  <class 'dict'> 1737
    ##
    strList = []
    with open(menStrDir, 'r', encoding="utf-8") as fr:
        for line in fr.readlines():
            if line.strip():
                strList.append(line.strip())
    fr.close()
    print('strList: ', type(strList), len(strList)) # strList:  <class 'list'> 2544
    ##
    #assert len(pageid2embedid) == len(strList)
    for pageid, embedid in pageid2embedid.items():
        embedid2str[embedid] = strList[pageid]
    strs_of_embeddings = [embedid2str[i] for i in range(len(embedid2str.keys()))]

    assert len(pageid2embedid) == len(strs_of_embeddings)
    return strs_of_embeddings



def top_sim_from_embeddings(obj_embedding, candidate_embeddings, candidate_strs, top_n, ):
    '''
    用embed直接比较，找top-n个最相似的，
    '''
    distances = cosine_similarity(obj_embedding, candidate_embeddings)
    keywords = [candidate_strs[index] for index in distances.argsort()[0][-top_n:]]
    return keywords


def cluster_from_embeddings(corpus0, vectors0, n_clusters, n_components, isNeglectUNK,clusImgDir):
    '''
    用embed，用K-means进行聚类
    可以在聚类之前先用PCA进行降维（必要）
    '''

    # 去躁（不考虑那些含有<unk>的词）
    i_list = [i for i in range(len(corpus0)) if corpus0[i].find('<unk>')==-1]
    corpus = [corpus0[i] for i in i_list]
    vectors = np.zeros((len(i_list), vectors0.shape[1]), dtype=np.float32)
    for j in range(len(i_list)):
        vectors[j] = vectors0[i_list[j]]
    print('corpus', len(corpus))
    print('vectors', vectors.shape)
    print(corpus)
    print(vectors)

    ###
    pca = PCA(n_components=n_components)  # （“保留多少信息的比例”，这个确实是有用的。OK。） # n_components若取整数：降到几维；若取小数(<1)：保留多少信息。
    vectors_ = pca.fit_transform(vectors)  #
    print('vectors_', vectors_.shape)
    #####
    km = KMeans(n_clusters=n_clusters)
    y_ = km.fit_predict(vectors_)  # 聚类
    # 返回值 labels : ndarray of shape (n_samples,) 就是Index of the cluster each sample belongs to.

    # 输出结果
    clusIndex_strList = {}
    clusIndexList = list(set(list(y_)))
    for ci in clusIndexList:
        clusIndex_strList[ci] = []
    for i in range(len(y_)):
        clusIndex_strList[y_[i]].append(corpus[i])
    for key,val in clusIndex_strList.items():
        print(key, ' ::: ', val)

    '''
    # 画图展示
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.scatter(vectors_[:, 0], vectors_[:, 1], c=y_)  # 将点画在图上
    for i in range(len(corpus)):  # 给每个点进行标注
        plt.annotate(s=corpus[i], xy=(vectors_[:, 0][i], vectors_[:, 1][i]),
                     xytext=(vectors_[:, 0][i] + 0.1, vectors_[:, 1][i] + 0.1))
    #plt.show()
    plt.savefig(os.path.join(clusImgDir, '2 clus from 2 dim' + '.jpg'))
    '''

    return














''' 句子解析相关 '''
'''
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'D:\Python\stanford-corenlp-4.2.0')  # , logging_level=logging.NOTSET)
'''


def sentences_split_by_corenlp(nlp, sentences):
    '''
    还行，至少人名中的点不会被误认。
    :param nlp:
    :param sentences:
    :return:
    '''
    outputFormat = 'json'
    dpResult = nlp.annotate(sentences, properties={'annotators': 'ssplit', 'outputFormat': outputFormat, })
    sentsList = []
    try:
        for i in range(len(json.loads(dpResult)["sentences"])):
            sent_tokens = json.loads(dpResult)["sentences"][i]["tokens"]
            sent = ' '.join([sent_tokens[j]["word"] for j in range(len(sent_tokens))])
            sentsList.append(sent)
    except Exception as e:
        print(e)
        print('from【sentences_split_by_corenlp】')
        print(sentences)

    return sentsList

# sentences = 'Authored by: Tom J. Smith. The license are provided for you. You cannot modify the source code.'
# sentences_split_by_corenlp(nlp, sentences)
# nlp.close()


def identify_PERSON_ORGANIZATION_by_corenlp(nlp, sent):
    outputFormat = 'json'
    dpResult = nlp.annotate(sent, properties={'annotators': 'ner', 'outputFormat': outputFormat, })
    #print(dpResult)
    entsList = []
    try:
        if "entitymentions" in json.loads(dpResult)["sentences"][0].keys():
            ens = json.loads(dpResult)["sentences"][0]["entitymentions"]
            for i in range(len(ens)):
                if ens[i]["ner"] == "PERSON" or ens[i]["ner"] == "ORGANIZATION":
                    ent = ens[i]["text"]
                    entsList.append(ent)
    except Exception as e:
        print(e)
        print('from【identify_PERSON_ORGANIZATION_by_corenlp】')
        print(sent)

    return entsList

# # sentences = 'Authored by: Tom J. Smith The license are provided for David Saran and Free Software Foundation.'
# sentences = 'The license are provided for you.'
# # ['Tom J. Smith', 'David Saran', 'Free Software Foundation']
# print(identify_PERSON_ORGANIZATION_by_corenlp(nlp, sentences))
# nlp.close()



def extract_entity_mention(extractType, tokens, dpList, seedID, findedSet):
    if extractType=='compound':
        return extract_entity_mention_compound(dpList, seedID, findedSet)
    elif extractType=='nsubj':
        return extract_entity_mention_nsubj(dpList, seedID, findedSet)
    elif extractType=='VB':
        return extract_entity_mention_VB(tokens, seedID, findedSet)

def extract_entity_mention_compound(dpList, seedID, findedSet):
    '''
    compound联系的两边，若多次连接 就都连成一个，
    '''
    dtypes = [
        'compound',
    ]
    for dp in dpList:
        # if dp['governor'] == seedID and dp['dep'].split(':')[0] in dtypes:
        if dp['dep'].split(':')[0] in dtypes: ###
            # findedSet.add(seedID)
            FG = False
            for i in range(len(findedSet)):
                tp = findedSet[i]
                if tp[-1] == dp['governor']:
                    findedSet[i].append(dp['dependent'])
                    FG = True
                    break
                if tp[0] == dp['dependent']:
                    findedSet[i].insert(0, dp['governor'])
                    FG = True
                    break
            if not FG:
                findedSet.append([dp['governor'], dp['dependent']])
    return findedSet

def extract_entity_mention_nsubj(dpList, seedID, findedSet):
    '''
    nsubj，含有nsubj的关系，只需要它指向的那一边,（即每次一个单词）
    '''
    dtypes = [
        'nsubj',
    ]
    for dp in dpList:
        # if dp['governor'] == seedID and dp['dep'].split(':')[0] in dtypes:
        if dp['dep'].find(dtypes[0]) > -1:
            # findedSet.add(seedID)
            FG = False
            for i in range(len(findedSet)):
                if dp['dependent'] in findedSet[i]:
                    FG = True
                    break
            if not FG:
                findedSet.append([dp['dependent']])
    print(findedSet)
    return findedSet


def extract_entity_mention_VB(tokens, seedID, findedSet):
    '''
    这些动词词性的单词,（即每次一个单词）
    后面看要不要将word进行词性还原？？
    '''
    dtypes = [
        'VB', 'VBD', 'VBG', 'VBP', 'VBZ', 'VBN',
    ]
    for tk in tokens:
        if tk['pos'] in dtypes:
            findedSet.append([tk['index']])
    findedSet = get_unique_lists_in_list(findedSet, isInt=True)
    print(findedSet)
    return findedSet



############################################################

def extract_its_compound(dpList, findedSet):
    '''
    compound联系的两边，若多次连接 就都连成一个，
    '''
    '''更丰富些 名词前面的修饰成分det,amod, ''' ### 暂时不了。
    startID = findedSet[0]
    while True:
        FG = False
        for dp in dpList:
            if dp['governor'] == startID and dp['dep'].split(':')[0] in ['compound']:
                startID = dp['dependent']
                findedSet.append(startID)
                FG = True
                break
        if not FG:
            break

    findedSet.reverse()
    return findedSet


def extract_its_hierac_nsubj(dpList, seedID,):
    findedSet = []

    # 找它的直接主语A  (动词的主语就一个)
    direct_nsubj = -1
    for dp in dpList:
        if dp['governor'] == seedID and dp['dep'].find('nsubj') > -1:
            direct_nsubj = dp['dependent']
            findedSet.append(extract_its_compound(dpList, [direct_nsubj]))
            break

    # A也可能同时是B的宾语 找B （是它的间接主语）
    if direct_nsubj != -1:
        mid_vb = -1
        for dp in dpList:
            if dp['dependent'] == direct_nsubj and dp['dep'].find('obj') > -1:
                mid_vb = dp['governor']
                break
        if mid_vb != -1:
            for dp in dpList:
                if dp['governor'] == mid_vb and dp['dep'].find('nsubj') > -1:
                    findedSet.append(extract_its_compound(dpList, [dp['dependent']]))
                    break

    # 它可能是一个宾语从句的谓语 找主句的主语 （是它的间接主语）
    mid_vb = -1
    for dp in dpList:
        if dp['dependent'] == seedID and dp['dep'].split(':')[0] in ['ccomp', 'xcomp']:
            mid_vb = dp['governor']
            break
    if mid_vb != -1:
        for dp in dpList:
            if dp['governor'] == mid_vb and dp['dep'].find('nsubj') > -1:
                findedSet.append(extract_its_compound(dpList, [dp['dependent']]))
                break

    return findedSet


def extract_its_hierac_obj(dpList, seedID,):
    findedSet = []
    # 找它的直接宾语 （宾语可能并列好几个）
    for dp in dpList:
        if dp['governor'] == seedID and dp['dep'].find('obj') > -1:
            findedSet.append(extract_its_compound(dpList, [dp['dependent']]))
    # 找它的并列谓语的宾语
    mid_vb = -1
    for dp in dpList:
        if dp['dependent'] == seedID and dp['dep'].split(':')[0] in ['conj']: # :and, :or
            mid_vb = dp['governor']
            break
    if mid_vb != -1:
        for dp in dpList:
            if dp['governor'] == mid_vb and dp['dep'].find('obj') > -1:
                findedSet.append(extract_its_compound(dpList, [dp['dependent']]))

    return findedSet


def extrac_its_hierac_mod(dpList, seedID,):
    findedSet = []

    # 找它的直接情态动词/情态副词 (若干个)
    for dp in dpList:
        if dp['governor'] == seedID and dp['dep'].split(':')[0] in ['aux', 'advmod']:
            findedSet.append(extract_its_compound(dpList, [dp['dependent']]))

    # 间接的直接情态动词/情态副词 （若干个）
    mid_vb = -1
    for dp in dpList:
        if dp['dependent'] == seedID and dp['dep'].split(':')[0] in ['ccomp', 'xcomp']:
            mid_vb = dp['governor']
            break
    if mid_vb != -1:
        for dp in dpList:
            if dp['governor'] == mid_vb and dp['dep'].split(':')[0] in ['aux', 'advmod']:
                findedSet.append(extract_its_compound(dpList, [dp['dependent']]))

    # 找它的并列谓语的情态动词/情态副词 （若干个）
    mid_vb = -1
    for dp in dpList:
        if dp['dependent'] == seedID and dp['dep'].split(':')[0] in ['conj']:  # :and, :or
            mid_vb = dp['governor']
            break
    if mid_vb != -1:
        for dp in dpList:
            if dp['governor'] == mid_vb and dp['dep'].split(':')[0] in ['aux', 'advmod']:
                findedSet.append(extract_its_compound(dpList, [dp['dependent']]))


    return findedSet


def extract_its_hierac_cond(dpList, seedID,):
    findedSet = []

    # 找它的条件句的谓语 （可能多个）
    for dp in dpList:
        if dp['governor'] == seedID and dp['dep'].split(':')[0] in ['acl', 'advcl']:
            findedSet.append(extract_its_compound(dpList, [dp['dependent']]))

    ## "as long as"
    mid_conj_list = []
    for dp in dpList:
        if dp['governor'] == seedID and dp['dep'].split(':')[0] in ['advmod']: # "long"
            mid_conj_list.append(dp['dependent'])
    for mid_conj in mid_conj_list:
        for dp in dpList:
            if dp['governor'] == mid_conj and dp['dep'].split(':')[0] in ['acl', 'advcl']:
                findedSet.append(extract_its_compound(dpList, [dp['dependent']]))

    return findedSet





def get_words_from_ids(findedIDSet,tokens):
    got_wordlist = []

    for kid in findedIDSet:
        for tk in tokens:
            if tk['index'] == kid:
                got_wordlist.append(tk['word'])
                break

    return got_wordlist






# 测试一下
'''
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'D:\Python\stanford-corenlp-4.2.0', logging_level=logging.NOTSET)

sent = 'This License does not grant permission to use the trade file names, trademarks, service marks, or product file names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.'

outputFormat = 'json'
dpResult = nlp.annotate(sent, properties={'annotators': 'depparse', 'outputFormat': outputFormat, })
enhancedPlusPlusDependencies = json.loads(dpResult)["sentences"][0]["enhancedPlusPlusDependencies"]
tokens = json.loads(dpResult)["sentences"][0]["tokens"]
print(tokens)

findedIDSet = extract_entity_mention(extractType='VB', tokens=tokens, dpList=enhancedPlusPlusDependencies, seedID=0, findedSet=[])
for tp in findedIDSet:
    tp.reverse() ##
    phrase = get_words_from_ids(tp, tokens)
    print(phrase)

nlp.close()
'''

'''
from nltk.tree import Tree
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('model/stanford-corenlp-4.2.0')
sqc = 'subject to the terms and conditions of this license , each contributor hereby grants to you a perpetual , worldwide , non-exclusive , no-charge , royalty-free , irrevocable copyright license to reproduce , prepare derivative works of , publicly display , publicly perform , sublicense , and distribute the work and such derivative works in source or object form .'
zhi = nlp.parse(sqc)
tree = Tree.fromstring(zhi)
tree.draw()
'''


'''
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'D:\Python\stanford-corenlp-4.2.0')

sent = 'This License does not grant permission to use the trade file names, trademarks, service marks, or product file names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.'
#sent2 = 'What college did Dikembe Mutombo play basketball for ?'

outputFormat = 'json'
dpResult = nlp.annotate(sent, properties={'annotators': 'depparse', 'outputFormat': outputFormat, })
enhancedPlusPlusDependencies = json.loads(dpResult)["sentences"][0]["enhancedPlusPlusDependencies"]
tokens = json.loads(dpResult)["sentences"][0]["tokens"]

print(json.loads(dpResult)["sentences"][0].keys())
#print(json.loads(dpResult)["sentences"][0]["parse"])
#print(nlp.parse(sent2))
print(tokens)
for a in enhancedPlusPlusDependencies:
    print(a)

nlp.close()
'''





'''
【条款细节提取的相关】
（与KnowPrompt的来回接口）

'''





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

                if line=='O' or line.startswith('B-') or line.startswith('I-'):
                    continue

                assert len(line.split(' ')) == 2

                word = line.split(' ')[0].strip()

                if clean:
                    word = cleanText(line.split(' ')[0]).strip()  ### with清洗
                if not word:
                    continue

                words.append(word)
                labs.append(line.split(' ')[1])

    entities_chunks = get_chunks2(labs)
    return words, labs, entities_chunks



def getItsSequence(words, entity_chunk):
    SSList = ['.','!','?']
    '''
    beginIdx, endIdx = entity_chunk[1], entity_chunk[2] - 2
    for i in range(entity_chunk[1] - 1, -1, -1):  # backward ...
        beginIdx -= 1
        if words[beginIdx] in SSList:
            break
    for i in range(entity_chunk[2] - 1, len(words), 1):  # forward ...
        endIdx += 1
        if words[endIdx] in SSList:
            break
    '''
    beginIdx = 0
    endIdx = len(words)-1

    for i in range(entity_chunk[1] - 1, -1, -1):  # backward ...
        if words[i] in SSList:
            beginIdx = i+1
            break

    for i in range(entity_chunk[2] - 1, len(words), 1):  # forward ...
        if words[i] in SSList:
            endIdx = i
            break

    return beginIdx, ' '.join(words[beginIdx:endIdx+1]) # beginIdx是sqc的位置索引



def detect_condPos_in_sqc(sqc):
    condKeys = [# 认为关键词后面的是条件，前面的是主（如果有）
        "provided that",
        "condition ",
        "conditions ",
        "if ",
        "when ",
    ]
    for key in condKeys:
        if sqc.lower().find(key)>-1:
            return len(sqc[:sqc.lower().find(key)+1].split(' '))-1 # 返回在list里的ID
    return -1
def detect_condPosRange_in_sqc(sqc):
    condKeys = [# 认为关键词后面的是条件，前面的是主（如果有）
        "provided that",
        "condition ",
        "conditions ",
        "if ",
        "when ",
    ]
    for key in condKeys:
        if sqc.lower().find(key)>-1:
            inx = len(sqc[:sqc.lower().find(key)+1].split(' '))-1
            return [inx, inx+len(key.strip().split(' '))-1] # 闭闭
    return []
## print(detect_condPos_in_sqc('you can copy it .. vs provided that you retain them.'))


def get_tt_condInxs(words, labs, entities_chunks):
    '''

    :return:
    '''
    jj_condInxs = {}
    for j in range(23):
        jj_condInxs[j] = []

    jj_etChunkInx = {}
    etChunkInx_jj = {}
    sentStr_etChunks = {}
    entities_chunks_relative = []
    for etcInx, entity_chunk in enumerate(entities_chunks):
        j = int(entity_chunk[0])
        sent_beginIdx, sqc = getItsSequence(words, entity_chunk)
        chunk_beginIdx_relative = entity_chunk[1] - sent_beginIdx  # chunk在其sent里的位置索引
        entities_chunks_relative.append(chunk_beginIdx_relative)
        if j not in jj_etChunkInx.keys():
            jj_etChunkInx[j] = etcInx
            etChunkInx_jj[etcInx] = j
            if sqc in sentStr_etChunks.keys():
                sentStr_etChunks[sqc].append(etcInx)
            else:
                sentStr_etChunks[sqc] = [etcInx]

    for sqc, etChunks in sentStr_etChunks.items():
        condKeyID = detect_condPos_in_sqc(sqc)
        if condKeyID==-1:
            break

        # 找在关键词前面的
        tmp = []
        for etcInx in etChunks:
            if entities_chunks_relative[etcInx] < condKeyID:
                tmp.append(etcInx)
        # 用在后面的那些去赋给tmp里的们
        for etcInx_cond in etChunks:
            if entities_chunks_relative[etcInx_cond] > condKeyID:
                ###
                for etcInx_main in tmp:
                    jj_condInxs[etChunkInx_jj[etcInx_main]].append(etChunkInx_jj[etcInx_cond])

    return jj_condInxs, jj_etChunkInx





def get_type2id():
    type2id = {}
    for j, type in enumerate(term_config['term_list']):
        if j==23:
            continue
        type2id[type] = j
    return type2id

def get_id2type():
    id2type = {}
    for j, type in enumerate(term_config['term_list']):
        if j==23:
            continue
        id2type[j] = type
    return id2type



def get_id2rel(filename):
    with open(filename, 'r', encoding="utf-8") as fr:
        rel2id = json.load(fr)
    id2rel = dict([(v,k) for (k,v) in rel2id.items()])
    assert len(id2rel) == 6
    # print('id2rel', id2rel)
    return id2rel


def write_RE_file(data_list, filepath):
    with open(filepath, 'w', encoding="utf-8") as fw:
        for sample in data_list:
            fw.write(json.dumps(sample) + '\n')
    return


def write_BIO_file(words_list, labels_list, filepath):
    with open(filepath, 'w', encoding="utf-8") as fw:
        for i, (words, labels) in enumerate(zip(words_list, labels_list)):
            # 一条数据（one sent）
            for wd, la in zip(words, labels):
                fw.write(wd+' '+la+'\n')
            if i < len(words_list)-1:
                fw.write('\n')
    return


def write_BIO_file_tsv(words_list, labels_list, filepath):
    with open(filepath, 'w', encoding="utf-8") as fw:
        for words, labels in zip(words_list, labels_list):
            # 一条数据（one sent）
            for wd, la in zip(words, labels):
                fw.write(wd+'\t'+la+'\n')
            fw.write('\n')
    return



def get_type_from_etcPos(entities_chunks, posList):
    for i, entity_chunk in enumerate(entities_chunks):
        et_type = entity_chunk[0]
        if entity_chunk[1:3] == posList:
            return et_type
    return ""







'''
其他杂七杂八
'''
def complete_licenses_from_idxs():
    '''
    输入final.csv，
    把行名从数字换成许可证名字，把极性从数字换成字符串
    把结果写入新的文件tldr-licenses.csv
    :return:
    '''
    df = pd.read_csv("./data/label-original.csv")
    print(df) # 212*25
    print(df.columns)
    feats = ['Distribute', 'Modify', 'Commercial Use', 'Hold Liable',
       'Include Copyright', 'Include License', 'Sublicense', 'Use Trademark',
       'Private Use', 'Disclose Source', 'State Changes', 'Place Warranty',
       'Include Notice', 'Include Original', 'Give Credit',
       'Use Patent Claims', 'Rename', 'Relicense', 'Contact Author',
       'Include Install Instructions', 'Compensate for Damages',
       'Statically Link', 'Pay Above Use Threshold'] # *23
    df = df[feats]

    # 转换表
    num_atti = {
        1:'can', 2:'cannot', 3:'must', 0:'NOmentioned'
    }
    id_name = {}
    with open("./data/filter-exclude-list-forSpdx.txt", 'r', encoding='utf-8') as f:
        for li in f.readlines():
            id = li.strip().split(' ')[0]
            id = int(id)-1
            name = ' '.join(li.strip().split(' ')[2:])
            id_name[id] = name
    f.close()
    print(len(id_name), id_name)
    print(id_name.values())

    # 转换
    df.rename(index=id_name, inplace=True)
    print(df)

    TMP = []
    for row in df.itertuples():
        tmp = []
        for num in row[1:]:
            atti = num_atti[num]
            tmp.append(atti)
        TMP.append(tmp)

    # 生成新的
    df2 = pd.DataFrame(TMP, columns=feats, index=id_name.values())
    print(df2)
    df2.to_csv("./data/tldr-licenses-forSpdx.csv",encoding="utf-8")


    return


def reform_custom_template_csv():
    index = [
        'Distribute','Modify','Commercial Use','Hold Liable','Include Copyright',
        'Include License', 'Sublicense', 'Use Trademark', 'Private Use', 'Disclose Source',
        'State Changes', 'Place Warranty', 'Include Notice', 'Include Original', 'Give Credit',
        'Use Patent Claims', 'Rename', 'Relicense', 'Contact Author', 'Include Install Instructions',
        'Compensate for Damages', 'Statically Link', 'Pay Above Use Threshold'
    ]
    columns = ['template']
    values = [
        'distribute original or modified derivative works',
        'modify the software and create derivatives',
        'use the software for commercial purposes',
        'hold the author responsible for subsequent impacts',
        'retain the copyright notice in all copies or substantial uses of the work',

        'include the full text of license in modified software',
        'incorporate the work into something that has a more restrictive license',
        'use contributors\' names or trademarks or logos',
        'use or modify software freely without distributing it',
        'disclose your source code when you distribute the software and make the source for the library available',

        'state significant changes made to software',
        'place warranty on the software licensed',
        'include that NOTICE when you distribute if the library has a NOTICE file with attribution notes',
        'distribute copies of the original software or instructions to obtain copies with the software',
        'give explicit credit or acknowledgement to the author with the software',

        'practice patent claims of contributors to the code',
        'change software name as to not misrepresent them as the original software',
        'add other licenses with the software',
        'get permission from author or contact the author about the module you are using',
        'include the installation information necessary to modify and reinstall the software',

        'compensate the author for any damages cased by your work',
        'have the library compiled into the program linked at compile time rather than runtime',
        'pay the licensor after a certain amount of use',
    ]

    df = pd.DataFrame(values, index=index, columns=columns)
    df.to_csv('./data/custom_tldr_template.csv')

    df2 = pd.read_csv('./data/custom_tldr_template.csv')
    print(df2)
    print(df2.dtypes)
    print(df2.index)
    print(df2.columns)

def read_custom_template():
    termContent_template = {}
    with open('./data/custom_tldr_template.csv', 'r', encoding="utf-8") as fr:
        for line in fr.readlines():
            if line:
                termContent_template[line.strip().split(',')[0]] = line.strip().split(',')[1]
    fr.close()
    return termContent_template



###complete_licenses_from_idxs()
# reform_custom_template_csv()

''' 画图相关 '''
'''

'''
import matplotlib.pyplot as plt


# 定义画散点图的函数
def draw_scatter(n, dataX, dataY, labelX, labelY, title,imgDir):
    # 创建画图窗口
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax1 = fig.add_subplot(1, 1, 1)
    # 设置标题
    ax1.set_title(title)
    # 设置横坐标名称
    ax1.set_xlabel(labelX)
    # 设置纵坐标名称
    ax1.set_ylabel(labelY)

    # 画散点图
    ax1.scatter(dataX, dataY, s=5, c='k', marker='.')

    # 调整横坐标的上下界
    plt.xlim(xmax=n, xmin=0)
    # 保存图片
    plt.savefig(os.path.join(imgDir, title+'.jpg'))
    # 显示
    #plt.show()

    return


def draw_scatter_doubleColor(n, dataX1, dataY1, dataX2, dataY2, labelX, labelY):
    # 创建画图窗口
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax1 = fig.add_subplot(1, 1, 1)
    # 设置标题
    ax1.set_title('pro size analysis (has CL or not)')
    # 设置横坐标名称
    ax1.set_xlabel(labelX)
    # 设置纵坐标名称
    ax1.set_ylabel(labelY)

    # 画散点图
    ax1.scatter(dataX1, dataY1, s=5, c='k', marker='.')
    ax1.scatter(dataX2, dataY2, s=5, c='b', marker='.')

    # 调整横坐标的上下界
    plt.xlim(xmax=n, xmin=0)
    # 保存图片
    plt.savefig(labelY+'.jpg')
    # 显示
    plt.show()

    return



def draw_hist(dataL, bins, rangeUpLimit, xticksNum,labelX, labelY, title,imgDir):
    plt.figure()

    # 直方图关键操作
    plt.hist(dataL, bins=bins, range=(0,rangeUpLimit))
    plt.xticks(np.arange(0, rangeUpLimit, int(rangeUpLimit/xticksNum)))


    # 网格线，更好看
    plt.grid(alpha=0.5, linestyle='-.')

    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.title(title)
    plt.savefig(os.path.join(imgDir, title + '.jpg'))
    #plt.show()
    return




import codecs
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import graphviz

def to_graphviz(tree, filename, shape='circle', graph='digraph'):
    """Exports the tree in the dot format of the graphviz software"""
    '''
    给treelib中tree对象的
    '''

    nodes, connections = [], []
    if tree.nodes:

        for n in tree.expand_tree(mode=tree.WIDTH):
            nid = tree[n].identifier
            state = '{0} [ shape={1}]'.format(
                nid, shape)
            nodes.append(state)

            for c in tree.children(nid):
                cid = c.identifier
                connections.append('{0} -> {1}'.format(nid, cid))

    # write nodes and connections to dot format
    is_plain_file = filename is not None
    if is_plain_file:
        f = codecs.open(filename, 'w', 'utf-8')
    else:
        f = StringIO()

    f.write(graph + ' tree {\n')
    for n in nodes:
        f.write('\t' + n + '\n')

    if len(connections) > 0:
        f.write('\n')

    for c in connections:
        f.write('\t' + c + '\n')

    f.write('}')

    if not is_plain_file:
        print(f.getvalue())

    f.close()


    '''
    显示
    '''
    with(open(filename)) as f:
        dot_graph = f.read()
    dot = graphviz.Source(dot_graph)
    dot.view()


def to_graphviz_for_lr(tree, nid_filepath, filename, fixable_list, incom_list, repair_list, shape='circle', graph='digraph'):
    '''
    https://mathpretty.com/13259.html
    :param tree:
    :param filename:
    :param shape:
    :param graph:
    :return:
    '''
    # def getShortPath(nid):
    #     if nid==1:
    #         return 'root'
    #     return '\\'.join(nid_filepath[nid][len('D:\Python\OSSL2\evaluation_projects/repos_simpled_200/'):].split('\\')[2:])


    nodes, connections = [], []
    if tree.nodes:

        for n in tree.expand_tree(mode=tree.WIDTH):
            # 每个结点的状态
            nid = tree[n].identifier

            ###
            color = 'black'
            if nid in repair_list:
                color = 'green'
            elif nid in incom_list:
                color = 'red'
            elif nid in fixable_list:
                color = 'yellow'

            if color!='black':
                state = '{0} [ shape={1}, style=filled, color={2}]'.format(nid, shape, color) #
            else:
                state = '{0} [ shape={1}, color={2}]'.format(nid, shape, color) #

            nodes.append(state)

            # 找到边
            for c in tree.children(nid):
                cid = c.identifier
                connections.append('{0} -> {1}'.format(nid, cid)) #


    # 输出到.dot文件
    # write nodes and connections to dot format
    is_plain_file = filename is not None
    if is_plain_file:
        f = codecs.open(filename, 'w', 'utf-8')
    else:
        f = StringIO()

    f.write(graph + ' tree {\n')
    for n in nodes:
        f.write('\t' + n + '\n')

    if len(connections) > 0:
        f.write('\n')

    for c in connections:
        f.write('\t' + c + '\n')

    f.write('}')

    if not is_plain_file:
        print(f.getvalue())

    f.close()

    '''
    用graphviz来显示树状图。
    '''
    with(open(filename)) as f:
        dot_graph = f.read()
    dot = graphviz.Source(dot_graph)
    dot.view(filename=filename.split('/')[-1][:-len('.dot')], directory=filename[:-len(filename.split('/')[-1])] )







