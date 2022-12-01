# -*- coding:utf-8 -*-
'''



'''

import re
import os

import utils

rootDir = os.path.dirname(os.path.abspath(__file__))
unDir = os.path.join(os.path.dirname(rootDir), 'repos') #####


outputDir000 = rootDir + '/output/'
outputDir = rootDir + '/output/pros/'
DIR = outputDir


licenseDir = os.path.dirname(os.path.abspath(__file__))+'/data/licenses'



REGEXP = [
    re.compile(r'^import (.+)$'),
    re.compile(r'^from ((?!\.+).*?) import (?:.*)$')
]


def checkPackageImport2(filepath):
    try:
        imports = []
        with open(filepath, 'r', encoding="utf-8") as fr:
            for line in fr.readlines():
                if "import " in line:
                    if "from" in line:
                        match = REGEXP[1].match(line.strip())
                        if match:
                            name = match.groups(0)[0]
                            for im in name.partition(' as ')[0].partition(','):
                                nm = im.strip().partition('.')[0].strip()
                                if len(nm) > 1:
                                    imports.append(nm)
                    else:
                        match = REGEXP[0].match(line.strip())
                        if match:
                            name = match.groups(0)[0]
                            for im in name.partition(' as ')[0].partition(','):
                                nm = im.strip().partition('.')[0].strip()
                                if len(nm) > 1:
                                    imports.append(nm)
        return list(set(imports))
    except Exception:
        print(filepath)
        return []




from treelib import Tree, Node
tree = Tree()
nid_filepath = {}
nid_textNeedTE = {}
nid_matchedLnameList = {}

license_check, _ = utils.get_licenseNameList1(os.path.dirname(os.path.abspath(__file__))+'/data/filter-exclude-list.txt')
licenseNameList = utils.get_licenseNameList2(licenseDir)
licenseTextDict = utils.get_licenseTextDict2(licenseDir)



def add_node(parent, ziji, ziji_content, checked=True):
    '''

    :param parent:
    :param ziji:
    :param ziji_content:
    :param checked:
    :return:
    '''
    tree.create_node(parent=parent, identifier=ziji, tag=ziji_content)
    return ziji


def update_tag(nid, tag):

    # tree.update_node(nid=nid, attrs={'tag':tag}) ## （这个函数似乎没起作用，，）
    tree[nid].tag = tag

    print("更新PL")
    print(nid, tag)
    print("现在的PL为：")
    print(tree[nid].tag)

    return



IDsave = 0
def gen_id():
    global IDsave
    IDsave += 1
    return IDsave
def rmv_id():
    global IDsave
    IDsave -= 1
    return IDsave


def checkPro(dir, parent, fg):
    '''

    :param dir:
    :param parent:
    :return:
    '''

    '''
    （目标项目的存放路径）
    '''
    repoDir = os.path.join(unDir,dir)

    dir_prt = parent
    pac_prt = parent

    print(repoDir) ###

    # （先看file后看py）（对结果有影响。）
    FileList = []
    for dd in os.listdir(repoDir):
        dd_path = os.path.join(repoDir, dd)
        if os.path.isfile(dd_path) and not dd_path.endswith(".py"):
            FileList.append(dd)
    for dd in os.listdir(repoDir):
        dd_path = os.path.join(repoDir, dd)
        if os.path.isfile(dd_path) and dd_path.endswith(".py"):
            FileList.append(dd)

    #####
    for dd in FileList:
        dd_path = os.path.join(repoDir, dd)
        print(dd_path)  ###

        text = ''
        # if not dd_path.endswith(".py") and utils.checkLicenseFileName(dd):
        if utils.checkLicenseFileName(dd):
            text = utils.read_text(dd_path)
        if text and utils.check_text_for_licenseWords(text, license_check, licenseNameList):
            '''
            matchedLnameList0 = utils.match_availableText_for_possible_refLicenseTexts(text, licenseTextDict)
            refText, matchedLnameList1 = utils.add_possible_refLicenseTexts(licenseNameList, text, './data/licenses')
            text += refText
            '''
            matchedLnameList0 = utils.match_availableText_for_possible_refLicenseTexts(text, licenseTextDict)
            refText, matchedLnameList1 = utils.add_possible_refLicenseTexts(licenseNameList, text, licenseDir)
            textNeedTE = True
            if matchedLnameList0:
                textNeedTE = False

            if parent == 1 and fg != -1:
                # （PL若多个文件 认为是互相补充的 故合成一份text（一个节点））
                update_tag(nid=fg, tag=tree[fg].tag + text) # setup.py和__pkginfo__.py也可能会进入这里
                '''
                
                '''
                if nid_textNeedTE[fg] or textNeedTE:
                    nid_textNeedTE[fg] = True
                else:
                    nid_textNeedTE[fg] = False
                # if not nid_textNeedTE[fg] or not textNeedTE:
                #     nid_textNeedTE[fg] = False

            else:
                file_id = gen_id()
                dir_prt = add_node(parent, file_id, text)
                nid_filepath[file_id] = repoDir  ###
                nid_matchedLnameList[file_id] = matchedLnameList0 + matchedLnameList1
                nid_textNeedTE[file_id] = textNeedTE
                pac_prt = dir_prt
                print('pac_prt=',pac_prt)

                if parent == 1:
                    fg = file_id

        if dd_path.endswith(".py"):
            pac_prt_py = int(pac_prt) # (同地址赋值;引用赋值)
            text = utils.extract_comments_in_pyFile(dd_path)
            if text and utils.check_text_for_licenseWords(text, license_check, licenseNameList):
                matchedLnameList0 = utils.match_availableText_for_possible_refLicenseTexts(text, licenseTextDict)
                refText, matchedLnameList1 = utils.add_possible_refLicenseTexts(licenseNameList, text, licenseDir)
                textNeedTE = True
                if matchedLnameList0:
                    textNeedTE = False

                if (dd=='setup.py' or dd=='__pkginfo__.py') and parent == 1 and fg != -1:
                    # （setup.py可能也加进去（一般只涉及到PL））
                    update_tag(nid=fg, tag=tree[fg].tag + text)
                    '''
                    if nid_textNeedTE[fg] or textNeedTE:
                        nid_textNeedTE[fg] = True
                    else:
                        nid_textNeedTE[fg] = False
                    '''
                    if not nid_textNeedTE[fg] or not textNeedTE:
                        nid_textNeedTE[fg] = False

                else:
                    inline_id = gen_id()
                    pac_prt_py = add_node(pac_prt, inline_id, text)
                    nid_filepath[inline_id] = os.path.join(repoDir, dd)  ###
                    nid_matchedLnameList[inline_id] = matchedLnameList0 + matchedLnameList1
                    nid_textNeedTE[inline_id] = textNeedTE

            packages = checkPackageImport2(dd_path)
            for aa in packages:
                if aa in library_license.keys():
                    ll = library_license[aa]  #
                    print('        ', aa, ':::::', ll)
                    # (找到ll对应的text)
                    refText, matchedLnameList1 = utils.add_possible_refLicenseTexts(licenseNameList, ll, licenseDir)
                    text = ''
                    #if text:  # （能在SPDX找到的才算进去吧，，）
                    if matchedLnameList1:
                        ll_id = gen_id()
                        add_node(pac_prt_py, ll_id, text)
                        nid_filepath[ll_id] = os.path.join(repoDir, dd) + ':' + aa  ###
                        nid_matchedLnameList[ll_id] = [] + matchedLnameList1
                        nid_textNeedTE[ll_id] = False



    for dd in os.listdir(repoDir):
        dd_path = os.path.join(repoDir,dd)

        if os.path.isdir(dd_path):
            # print(dd_path)
            '''
            递归！
            '''
            checkPro(dd_path, dir_prt, fg)


    return


def check_PL(repo):
    repoDir = os.path.join(unDir, repo)
    repoDir = os.path.join(repoDir, os.listdir(repoDir)[0])
    '''
    按从GitHub下载的文件夹 第二层才是正经文件
    '''


    for file in os.listdir(repoDir):
        itsCompletePath = os.path.join(repoDir, file)
        print('check_PL:', itsCompletePath)

        if os.path.isfile(itsCompletePath):

            text = ''
            if utils.checkLicenseFileName(file):
                text = utils.read_text(itsCompletePath)

            if text:
                '''
                
                return True
                '''
                if utils.check_text_for_licenseWords(text, license_check, licenseNameList):
                    return True

    nid_filepath[-1] = repoDir
    nid_matchedLnameList[-1] = []
    nid_textNeedTE[-1] = False

    return False




'''
【这里是调用入口 从licenseRepair类那里】
'''
def get_license_tree(repo):
    init()
    '''
    
    '''
    global tree
    tree = Tree()

    global nid_filepath
    nid_filepath = {}
    global nid_textNeedTE
    nid_textNeedTE = {}
    global nid_matchedLnameList
    nid_matchedLnameList = {}


    global IDsave
    IDsave = 0


    #print(license_check)
    #print(licenseNameList)

    add_node(tree.root, gen_id(), 'root', checked=False)
    checkPro(repo, 1, -1)

    hasPL = check_PL(repo)

    return tree, nid_filepath, hasPL, nid_textNeedTE, nid_matchedLnameList




library_license = {}

def init():
    with open(outputDir000 + "library_license.txt", 'r', encoding='utf-8')as fr:
        for line in fr.readlines():
            if line.strip():
                line = line.strip()
                library_license[line.split(" ::::: ")[0]] = line.split(" ::::: ")[1]
    fr.close()
    #print(library_license)
    #print("library_license: " + str(len(library_license)))

