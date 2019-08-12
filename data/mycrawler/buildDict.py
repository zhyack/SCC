#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs

f = codecs.open('data/train.comments', 'r', 'UTF-8')
d = {}
for line in f.readlines():
    for word in line.split():
        if word not in d:
            d[word]=0
        d[word]+=1
f.close()
f = codecs.open('data/text.dict', 'w', 'UTF-8')
for w in d.keys():
    if d[w]>4:
        f.write(w+'\n')
f.close()


f = codecs.open('data/train.comments', 'r', 'UTF-8')
comments = f.readlines()
f.close()

classes = [[] for _ in range(len(comments))]
pred_labels_mapping = [1, 2, 4, 5]
for k in range(4):
    fcat_classes = codecs.open('data/train.comments.pred_labels_%d'%(k), 'r', 'UTF-8')
    cc = pred_labels_mapping[k]
    for i, line in enumerate(fcat_classes):
        if line[0]=='1':
            classes[i].append(cc)
    fcat_classes.close()
fcat_classes = codecs.open('data/train.comments.pseudoLabels', 'r', 'UTF-8')
for i, line in enumerate(fcat_classes):
    pp = eval(line.split('||||')[1])[1]
    if pp>0:
        classes[i].append(3)
fcat_classes.close()

# for c in range(1,6):
#     d = {}
#     for i, line in enumerate(comments):
#         if c in classes[i]:
#             for word in line.split():
#                 if word not in d:
#                     d[word]=0
#                 d[word]+=1
#     f = codecs.open('data/text_%d.dict'%(c), 'w', 'UTF-8')
#     if len(d)>20000:
#         thres = 1
#     else:
#         thres = 0
#     for w in d.keys():
#         if d[w]>thres:
#             f.write(w+'\n')
#     f.close()

import os
from pickle import Pickler, Unpickler

def loadTrainExamples(examplesFile):
    if not os.path.isfile(examplesFile):
        raise Exception("File with trainExamples not found. %s"%(examplesFile))
    else:
        print("File with trainExamples found. Read it. %s"%(examplesFile))
        with open(examplesFile, "rb") as f:
            ExamplesHistory = Unpickler(f).load()
        f.closed
        return ExamplesHistory

for c in range(1, 6):
    d = {}
    train_replays = loadTrainExamples('data/train_%d.pickle'%(c-1))
    boards, moves, vs, valids, texts = list(zip(*train_replays))
    for text in texts:
        text = text.split('@\t@')[1]
        for word in text.split():
            if word not in d:
                d[word]=0
            d[word]+=1
    f = codecs.open('data/text_%d.dict'%(c), 'w', 'UTF-8')
    if len(d)>20000:
        thres = 1
    else:
        thres = 0
    for w in d.keys():
        if d[w]>thres:
            f.write(w+'\n')
    f.close()
