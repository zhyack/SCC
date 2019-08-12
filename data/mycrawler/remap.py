#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs

from pickle import Pickler, Unpickler
import os

def loadExamples(examplesFile):
    with open(examplesFile, "rb") as f:
        ExamplesHistory = Unpickler(f).load()
    f.closed
    return ExamplesHistory


def saveExamples(trainExamples, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, filename), "wb+") as f:
        Pickler(f).dump(trainExamples)
    f.closed

def buildMap(es):
    ret = dict()
    for i, e in enumerate(es):
        e = list(e)
        le = e[-1].split('@\t@')
        assert(len(le)==2)
        cs = list([int(s)-1 for s in le[0].split()])
        ret[le[1]] = (cs, i)
    return ret

cats = ['test', 'train', 'valid']

for cat in cats:
    all_examples = loadExamples('data/%s.pickle'%(cat))
    m = buildMap(all_examples)
    for clss in range(3):
        fmap = codecs.open('data/map-%s-%d.txt'%(cat, clss), 'w', 'UTF-8')
        f = codecs.open('data/%s_%d.en'%(cat, clss), 'r', 'UTF-8')
        picked_examples = f.readlines()
        f.close()
        replays = []
        missCnt = 0
        for si, s in enumerate(picked_examples):
            print('Dealing with %s-%d-%d...'%(cat, clss, si))
            s = s.strip()
            if s in m:
                e = list(all_examples[m[s][1]])
            else:
                is_find = False
                for ss in m:
                    if ss.find(s)!=-1 and clss in m[ss][0]:
                        e = list(all_examples[m[ss][1]])
                        is_find = True
                        break
                if not is_find:
                    for ss in m:
                        if ss.find(s)!=-1:
                            e = list(all_examples[m[ss][1]])
                            is_find = True
                            break
                if not is_find:
                    for ss in m:
                        if ss.find(s[:len(s)//3])!=-1:
                            e = list(all_examples[m[ss][1]])
                            is_find = True
                            break
                if not is_find:
                    for ss in m:
                        if ss.find(s[len(s)//3:])!=-1:
                            e = list(all_examples[m[ss][1]])
                            is_find = True
                            break
                if not is_find:
                    missCnt += 1
                    if cat == 'test':
                        # raise Exception("Cannot Find %s for %s-%d"%(s, cat, clss))
                        print("Cannot Find %s for %s-%d, missCnt:%d"%(s, cat, clss, missCnt))
                    else:
                        continue
            fmap.write(str(si)+'\n')
            e[-1]=str(clss+1)+'@\t@'+s
            replays.append(tuple(e))
        fmap.close()
        # saveExamples(replays, 'data/', '%s_%d.pickle'%(cat,clss))
    # for clss in range(3,5):
    #     replays = []
    #     for s in m:
    #         if clss in m[s][0]:
    #             e = list(all_examples[m[s][1]])
    #             e[-1]=str(clss+1)+'@\t@'+s
    #             replays.append(tuple(e))
    #     saveExamples(replays, 'data/', '%s_%d.pickle'%(cat,clss))