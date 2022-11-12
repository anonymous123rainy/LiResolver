import os
import random
from tgrocery import Grocery

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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

#####
def train():
    grocery = Grocery('ossl2_ac')


    datalist = []
    DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
    dataDir = DIR + "data/termEntityTagging/"
    for file in os.listdir(dataDir):
        words, labs, entities_chunks = get_entities(os.path.join(dataDir, file), clean=False)
        for ck in entities_chunks:
            text = ' '.join(words[ck[1]:ck[2]])
            tag = int(ck[0])  # 0-22
            datalist.append((tag, text))
    print('datalist: ', len(datalist))
    print(datalist[:5])

    random.shuffle(datalist)
    train_src = datalist[:int(len(datalist)/5*4)]
    test_src = datalist[int(len(datalist)/5*4):]
    print('train_src: ', len(train_src))
    print('test_src: ', len(test_src))

    grocery.train(train_src)

    test_score = grocery.test(test_src)
    print(test_score)
    print(test_score.accuracy_overall)
    test_score.show_result()


    grocery.save()

    return




def predict(text, ac_model):
    # new_grocery = Grocery('ossl2_ac')
    # new_grocery.load()

    pred = ac_model.predict(text)
    # print(pred.predicted_y)
    # print(pred.dec_values)
    return pred.predicted_y # label(int)


# train()
# predict('state changes')
