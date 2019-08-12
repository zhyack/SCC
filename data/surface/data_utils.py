#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import json

def save2json(d, pf):
    f = codecs.open(pf,'w','utf-8')
    f.write(json.dumps(d, ensure_ascii=False, indent=4))
    f.close()
def json2load(pf):
    f = codecs.open(pf,'r','utf-8')
    s = ''.join(f.readlines())
    d = json.loads(s)
    f.close()
    return d
