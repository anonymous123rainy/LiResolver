#coding=utf-8
import os
import json
import utils
import LicenseRepair
from LicenseDataset import Licensedataset
from treelib import Tree, Node

rootDir = os.path.dirname(os.path.abspath(__file__))
unDir = os.path.join(rootDir, 'repos')




# （先加载好corenlp）
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(os.path.join(rootDir, 'model', 'stanford-corenlp-4.2.0'))
# （加载ee5模型）
from EE5.LocateTerms.nermodel.ner_model import NERModel
from EE5.LocateTerms.nermodel.config import Config
ner_config_ee5 = Config()
ner_model_ee5 = NERModel(ner_config_ee5)
ner_model_ee5.build()
ner_model_ee5.restore_session(ner_config_ee5.dir_model)
print('【ee5 loaded. 】')
# （加载re模型）
from RE import re_predict
re_args, re_model = re_predict.load_re_model()
print('【re loaded. 】')
# ner_model_ee5 = None
# re_args = None
# re_model = None
# （加载ac模型）
from tgrocery import Grocery
ac_model = Grocery(os.path.dirname(os.path.abspath(__file__))+'/'+'AC/ossl2_ac')
ac_model.load()
print('【ac loaded. 】')
## 加载ld
ld = Licensedataset()
ld.load_licenses_from_csv(nlp, ld, ner_model_ee5, re_args, re_model, ac_model)
print('【ld loaded. 】')





repo = "XXXX"
lr, fg_hasPL, num_fixable, num_incom, num_repair, methods_repair \
        = LicenseRepair.runLicenseRepair(repo, nlp, ld, ner_model_ee5, re_args, re_model, ac_model)



# （关闭corenlp）
nlp.close()





