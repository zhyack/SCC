#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

url_format = "https://gameknot.com/list_annotated.pl?u=all&c=0&sb=0&rm=0&rn=0&rx=9999&sr=0&p=%d"


from crawler_utils import *
from data_utils import *
from import_list import *

socket.setdefaulttimeout(5)


import random
l = list(range(0, 308))
# random.shuffle(l)
ratings = []

for i in l:
    while True:
        errorcode, content = getPage(url_format%(i))
        if errorcode!=0:
            print('Failed %d.'%(i))
            continue
        if content.find('Annotated by')==-1:
            print('Failed %d.'%(i))
            continue
        pos = 0
        while(True):
            st = content.find('Annotated by', pos)
            if st==-1:
                break
            g = re.findall('\((\d+)\)', content[st:st+200])
            if len(g)>0:
                ratings.append(int(g[0]))
            pos = st+200
        print('Complete %d.'%(i))
        print(sum(ratings), len(ratings), sum(ratings)/len(ratings))
        break
f = codecs.open('stat.txt', 'w', 'UTF-8')
for r in ratings:
    f.write(str(r)+'\n')
f.close()
