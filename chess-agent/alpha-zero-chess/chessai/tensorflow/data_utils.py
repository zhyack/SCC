#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import codecs

def buildDict(pfs, threshold=1, save2file=None):
    d = dict()
    for pf in pfs:
        f = codecs.open(pf, 'r', 'utf-8')
        ws = ''.join(f.readlines()).split()
        for w in ws:
            if w not in d:
                d[w]=0
            d[w]+=1
    if '<UNK>' not in d:
        d['<UNK>'] = 0
    for k in d.keys():
        if d[k]<threshold:
            if k == '<UNK>':
                continue
            d['<UNK>']+=d[k]
            del(d[k])
    rd = dictSort(d, bigfirst=True)
    if save2file:
        f = codecs.open(save2file, 'w', 'utf-8')
        for k in rd:
            f.write(k[0]+'\t'+str(k[1])+'\n')
        f.close()
    return d

def loadDict(pf):
    f = codecs.open(pf, 'r', 'UTF-8')
    lcnt = 0
    ret = dict()
    r_ret = dict()
    for l in f.readlines():
        w = l.strip()
        ret[w]=lcnt
        r_ret[lcnt]=w
        lcnt += 1
    if '<BOS>' not in ret:
        ret['<BOS>']=lcnt
        r_ret[lcnt]='<BOS>'
        lcnt += 1
    if '<EOS>' not in ret:
        ret['<EOS>']=lcnt
        r_ret[lcnt]='<EOS>'
        lcnt += 1
    if '<UNK>' not in ret:
        ret['<UNK>']=lcnt
        r_ret[lcnt]='<UNK>'
        lcnt += 1
    if '<PAD>' not in ret:
        ret['<PAD>']=lcnt
        r_ret[lcnt]='<PAD>'
        lcnt += 1
    return ret, r_ret




def getDecoderData(texts, text_dict, max_len=None, dtype=np.int32):
    ret = []
    ret_targets = []
    ret_len = []
    ret_mask = []
    def getData(s, text_dict, max_len):
        ret = []
        ndict = len(text_dict)
        ret.append(text_dict['<BOS>'])
        for w in s.split():
            if w not in text_dict:
                w = '<UNK>'
            ret.append(text_dict[w])
        nr = len(ret)
        if max_len==None:
            max_len=nr
        if max_len and nr > max_len:
            ret = ret[:max_len]
            nr = max_len
        ret_l = nr
        ret_t = ret[1:]+[text_dict['<EOS>']]
        ret_m = [1]*ret_l
        while(nr < max_len):
            ret.append(text_dict['<PAD>'])
            ret_t.append(text_dict['<PAD>'])
            nr += 1
            ret_m.append(0)
        return ret, ret_t, ret_l, ret_m
    for s in texts:
        x, x_t, x_l, x_m = getData(s, text_dict, max_len)
        ret.append(x)
        ret_targets.append(x_t)
        ret_len.append(x_l)
        ret_mask.append(x_m)
    ret_len = np.array(ret_len, dtype=np.int32)
    ret = np.array(ret, dtype=dtype)[:, :np.amax(ret_len)]
    ret_targets = np.array(ret_targets, dtype=dtype)[:, :np.amax(ret_len)]
    ret_mask = np.array(ret_mask, dtype=np.float32)[:, :np.amax(ret_len)]
    return ret, ret_targets, ret_len, ret_mask
    
def dataLogits2Seq(x, text_dict, calc_argmax=False):
    if calc_argmax:
        x = x.argmax(axis=-1)
    ret = ''
    for w in x:
        try:
            ret += text_dict[w]+' '
        except:
            pass
            # print(w)
    return ret