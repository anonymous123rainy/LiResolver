修订说明
==================
TextGrocery 是基于LibShortText项目(http://www.csie.ntu.edu.tw/~cjlin/libshorttext)的开源项目，仅支持Linux平台。
LibShortText项目原仅支持linux平台，但我已将其windows支持发布到github上，see:
    https://github.com/cosmichut/libshorttext-crossplatform

故本项目结合了以下三个项目：
- LibShortText项目
- TextGrocery原始项目，仅支持python2, 见 https://github.com/2shou/TextGrocery
- TextGrocery Python3项目（fork from 2shou) ,见 https://github.com/prashnts/TextGrocery

新的改动实现了以下目标:
- 同时支持windows(X64)和linux平台，无需修改代码。
- 支持Python3 (也同时支持python2)

在windows上安装(Linux 上也类似)
================================
1. go to root directory of this project, and Type

        python setup.py install
 
2. OR, if you want to do build and install ,then type 

        python setup.py make_and_install

    notice: under windows, you need  X64 Native Tools Command Prompt for VS2017 as following:

在windows上编译
=========================
follow the steps:

1. Open "X64 Native Tools Command Prompt for VS2017" comand line tools.
   also you can open a dos command window and set environment variables of VC++ like this, type

        ""C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64\vcvars64.bat""

   You may have to modify the above command according which version of VC++/VS or where it is installed.

2. change to root directory, and Type

        nmake -f Makefile.win clean liball

3. (Optional) To build 32-bit windows binaries, you must
	(1) Setup "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\vcvars32.bat" instead of vcvars64.bat
	(2) Change CFLAGS in every Makefile.win: /D _WIN64 to /D _WIN32

        nmake -f Makefile.win clean liball

4.  go to the ../demo ,and copy the command in demo.sh, and paste to command line to run:

        python classify.py

    notice: it runs several steps but possible error in code/codec
    
Todo List
==============
- 使用PEP 8规范化python代码（特别是liblinear部分）
- 支持pip安装（会取个新名字，比如tgrocery-python3)
- continue original author 2shou's plan


Enjoy!

Justin  @github: https://github.com/cosmichut   @2017/10/12


==================原作者说明======================================

TextGrocery
===========

[![Build Status](https://travis-ci.org/2shou/TextGrocery.svg?branch=master)](https://travis-ci.org/2shou/TextGrocery)

一个高效易用的短文本分类工具，基于[LibLinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear)

TextGrocery整合了[结巴分词](https://github.com/fxsjy/jieba)作为默认的分词单元，以支持中文的短文本分类

性能
----

- 训练集：来自32个类别的4.8万条新闻标题
- 测试集：来自32个类别的1.6万条新闻标题
- 与[scikit-learn](https://github.com/scikit-learn/scikit-learn)的svm和朴素贝叶斯算法做横向对比

|         分类器            | 准确率    |  计算时间（秒）   |
|:------------------------:|:---------:|:--------------:|
|     scikit-learn(nb)     |   76.8%   |     134        |
|     scikit-learn(svm)    |   76.9%   |     121        |
|     **TextGrocery**      | **79.6%** |    **49**      |

示例代码
-------

```python
>>> from tgrocery import Grocery
# 新开张一个杂货铺，别忘了取名！
>>> grocery = Grocery('sample')
# 训练文本可以用列表传入
>>> train_src = [
    ('education', '名师指导托福语法技巧：名词的复数形式'),
    ('education', '中国高考成绩海外认可 是“狼来了”吗？'),
    ('sports', '图文：法网孟菲尔斯苦战进16强 孟菲尔斯怒吼'),
    ('sports', '四川丹棱举行全国长距登山挑战赛 近万人参与')
]
>>> grocery.train(train_src)
# 也可以用文件传入
>>> grocery.train('train_ch.txt')
# 保存模型
>>> grocery.save()
# 加载模型（名字和保存的一样）
>>> new_grocery = Grocery('sample')
>>> new_grocery.load()
# 预测
>>> new_grocery.predict('考生必读：新托福写作考试评分标准')
education
# 测试
>>> test_src = [
    ('education', '福建春季公务员考试报名18日截止 2月6日考试'),
    ('sports', '意甲首轮补赛交战记录:米兰客场8战不败国米10年连胜'),
]
>>> new_grocery.test(test_src)
# 准确率
0.5
# 同样可以用文本传入
>>> new_grocery.test('test_ch.txt')
# 自定义分词器
>>> custom_grocery = Grocery('custom', custom_tokenize=list)
```

更多示例: [sample/](sample/)

安装
----

    $ pip install tgrocery 

> 目前仅在Unix系统下测试通过