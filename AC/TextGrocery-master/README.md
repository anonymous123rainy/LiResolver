Preface for revise
==================
TextGrocery is based on LibShortText(http://www.csie.ntu.edu.tw/~cjlin/libshorttext) and support linux only.
LibShortText as well support linux only, but I have added windows support,see:
    https://github.com/cosmichut/libshorttext-crossplatform

current project based on 3 projects：
- LibShortText
- TextGrocery original，python2 only,see https://github.com/2shou/TextGrocery
- TextGrocery Python3 support（fork from 2shou) ,see https://github.com/prashnts/TextGrocery

current project achieve goal:
- support windows(X64) and linux both without need to change code
- support python3 as well as python2

Install on Windows(As well as under Linux)
==========================================
1. go to root directory of this project, and Type

        python setup.py install
 
2. OR, if you want to do build and install ,then type 

        python setup.py make_and_install

    notice: under windows, you need X64 Native Tools Command Prompt for VS2017 as "build on Windows"

Build on Windows
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
- python PEP 8 rule refactor
- support pip install , a new name like grocery-python3.
- continue original author 2shou's plan


Enjoy!

Justin  @github: https://github.com/cosmichut   @2017/10/12


==================original README.md======================================

TextGrocery
===========

[![Build Status](https://travis-ci.org/2shou/TextGrocery.svg?branch=master)](https://travis-ci.org/2shou/TextGrocery)

A simple, efficient short-text classification tool based on LibLinear

Embed with [jieba](https://github.com/fxsjy/jieba) as default tokenizer to support Chinese tokenize

Other languages: [更详细的中文文档](http://textgrocery.readthedocs.org/zh/latest/index.html)

Performance
-----------

- Train set: 48k news titles with 32 labels
- Test set: 16k news titles with 32 labels
- Compare with svm and naive-bayes of [scikit-learn](https://github.com/scikit-learn/scikit-learn)

|         Classifier       | Accuracy  |  Time cost(s)  |
|:------------------------:|:---------:|:--------------:|
|     scikit-learn(nb)     |   76.8%   |     134        |
|     scikit-learn(svm)    |   76.9%   |     121        |
|     **TextGrocery**      | **79.6%** |    **49**      |

Sample Code
-----------

```python
>>> from tgrocery import Grocery
# Create a grocery(don't forget to set a name)
>>> grocery = Grocery('sample')
# Train from list
>>> train_src = [
    ('education', 'Student debt to cost Britain billions within decades'),
    ('education', 'Chinese education for TV experiment'),
    ('sports', 'Middle East and Asia boost investment in top level sports'),
    ('sports', 'Summit Series look launches HBO Canada sports doc series: Mudhar')
]
>>> grocery.train(train_src)
# Or train from file
>>> grocery.train('train_ch.txt')
# Save model
>>> grocery.save()
# Load model(the same name as previous)
>>> new_grocery = Grocery('sample')
>>> new_grocery.load()
# Predict
>>> new_grocery.predict('Abbott government spends $8 million on higher education media blitz')
education
# Test from list
>>> test_src = [
    ('education', 'Abbott government spends $8 million on higher education media blitz'),
    ('sports', 'Middle East and Asia boost investment in top level sports'),
]
>>> new_grocery.test(test_src)
# Return Accuracy
1.0
# Or test from file
>>> new_grocery.test('test_ch.txt')
# Custom tokenize
>>> custom_grocery = Grocery('custom', custom_tokenize=list)
```

More examples: [sample/](sample/)

Install
-------

    $ pip install tgrocery

> Only test under Unix-based System
