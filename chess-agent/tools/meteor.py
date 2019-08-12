#coding: UTF-8
import os
import codecs

def evaluate(hyp, ref, num_ref=1):
    os.popen('cd ~/ICC/chess-agent/; python tools/make_meteor_file.py -i %s -o %s' %(ref, ref.replace('.txt', '.meteor')))
    outputs = ''.join(os.popen('cd ~/ICC/chess-agent/;java -Xmx2G -jar tools/meteor-1.5/meteor-1.5.jar %s %s -l en -norm -r %d' % (hyp, ref.replace('.txt', '.meteor'), num_ref)).readlines()[-1:])
    return outputs