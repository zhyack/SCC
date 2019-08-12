#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import urllib
import urllib.request
import socket
import codecs
import time
# socket.setdefaulttimeout(5)
import random
import eutils

headers = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0", "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv2.0.1) Gecko/20100101 Firefox/4.0.1", "Mozilla/5.0 (Windows NT 6.1; rv2.0.1) Gecko/20100101 Firefox/4.0.1", "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11", "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11"]


def makeReqGetRes(url):
    req = urllib.request.Request(url)
    req.add_header('User-Agent', headers[random.randint(0,len(headers)-1)])
    response = urllib.request.urlopen(req)
    return response

def getRealUrl(url):
    pre_url = ""
    while(url != pre_url):
        pre_url = url
        response = makeReqGetRes(url)
        url = response.geturl()
        time.sleep(float(random.randint(5,10))/10)
    return url


def getPage(url, saveto='tmp.html'):
    time.sleep(float(random.randint(5,20))/10)
    try:
        url = getRealUrl(url)
    except Exception:
        print('Failed to get url %s'%(url))
        return 2, None
    trycnt = 3
    while(trycnt>0):
        try:
            response = makeReqGetRes(url)
            content = eutils.econv(response.read())
        except IOError:
            print("Detected when crawling %s, retry now..."%(url))
            time.sleep(random.randint(1,2))
            trycnt -= 1
            continue
        break
    if trycnt==0:
        return 3, None
    ftmp = codecs.open(saveto,'w','UTF-8')
    ftmp.write(content)
    ftmp.close()
    return 0, content
