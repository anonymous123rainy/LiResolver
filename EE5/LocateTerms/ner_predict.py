'''
[许可证条款抽取]の测试数据の正式预测
'''
import os
from .nermodel.data_utils import CoNLLDataset
from .nermodel.ner_model import NERModel
from .nermodel.config import Config
import numpy as np



def printPred(config, model):
    #print("---------------------------------")

    #print(config.filename_dir_test)
    #print(config.filename_dir_pre)
    if not os.path.exists(config.filename_dir_pre):
        os.makedirs(config.filename_dir_pre)


    for root, dirs, files in os.walk(config.filename_dir_test):
        for file in files:
            #print(file)
            #print(files.index(file), '/', len(files))

            # fw = open(config.filename_dir_pre+file, 'w', encoding="utf-8")
            #
            # tmp = []
            # with open(config.filename_dir_test+file, 'r', encoding="utf-8")as fr:
            #     for line in fr.readlines():
            #         word = line.strip().split(' ')[0]
            #         if word == '.':
            #             tmp.append(word)
            #             words_raw = tmp
            #             ##
            #             preds = model.predict(words_raw)
            #             print(preds)
            #
            #
            #             for wd, pre in zip(words_raw, preds):
            #                 fw.write(wd + ' ' + pre + '\n')
            #                 print(wd, pre)
            #
            #             tmp.clear()
            #         else:
            #             tmp.append(word)

            # （因为这次输入一个句子 很可能很短没句号）

            words_raw = []
            with open(config.filename_dir_test + file, 'r', encoding="utf-8")as fr:
                for line in fr.readlines():
                    word = line.strip().split(' ')[0]
                    words_raw.append(word)

            preds = model.predict(words_raw)

            with open(config.filename_dir_pre+file, 'w', encoding="utf-8") as fw:
                for wd, pre in zip(words_raw, preds):
                    fw.write(wd + ' ' + pre + '\n')
                    # print(wd, pre)


            ## （OSError: [Errno 24] Too many open files）？？。。。
            fr.close()

            fw.close()

####################################################


def main(model):

    '''

    :return:
    '''
    config = Config()

    # build model
    '''
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)
    '''



    # 就这里每次都输出文件存储test/的预测结果吧
    printPred(config, model)


# # （加载ee5模型）
# from nermodel.ner_model import NERModel
# from nermodel.config import Config
# ner_config_ee5 = Config()
# ner_model_ee5 = NERModel(ner_config_ee5)
# ner_model_ee5.build()
# ner_model_ee5.restore_session(ner_config_ee5.dir_model)
#
# main(ner_model_ee5)
