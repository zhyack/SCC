#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def ngramCnt(t, max_n):
    s = set()
    cnt = 0
    wl = t.split()
    n = len(wl)
    for i in range(n-max_n+1):
        ngram = ' '.join(wl[i:i+max_n-1])
        s.add(ngram)
        cnt += 1
    return s, cnt


def corpus_diversity(texts, max_n=2):
    total_s = set()
    total_cnt = 0
    for t in texts:
        s, cnt = ngramCnt(t, max_n)
        total_s = total_s.union(s)
        total_cnt += cnt
    if total_cnt == 0:
        return 0.0
    else:
        return len(total_s)/float(total_cnt)