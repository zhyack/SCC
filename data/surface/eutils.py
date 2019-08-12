#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import chardet
CODEC={'unicode':['uni', 'Unicode'],\
'gbk':['936', 'cp936', 'ms936'],\
'utf_8':['U8', 'UTF', 'utf8', 'utf-8', 'UTF-8'],\
'gb2312':['chinese', 'csiso58gb231280', 'euc-cn', 'euccn', 'eucgb2312-cn', 'gb2312-1980', 'gb2312-80', 'iso-ir-58'],\
'ascii':['646','us-ascii'],\
'big5':['big5-tw','csbig5'],\
'big5hkscs':['big5-hkscs','hkscs'],\
'cp037':['IBM037', 'IBM039'],\
'cp273':['273', 'IBM273', 'csIBM273'],\
'cp424':['EBCDIC-CP-HE', 'IBM424'],\
'cp437':['437', 'IBM437'],\
'cp500':['EBCDIC-CP-BE', 'EBCDIC-CP-CH', 'IBM500'],\
'cp720':[],\
'cp737':[],\
'cp775':['IBM775'],\
'cp850':['850', 'IBM850'],\
'cp852':['852', 'IBM852'],\
'cp855':['855', 'IBM855'],\
'cp856':[],\
'cp857':['857', 'IBM857'],\
'cp858':['858', 'IBM858'],\
'cp860':['860', 'IBM860'],\
'cp861':['861', 'IBM861', 'CP-IS'],\
'cp862':['862', 'IBM862'],\
'cp863':['863', 'IBM863'],\
'cp864':['IBM864'],\
'cp865':['865', 'IBM865'],\
'cp866':['866', 'IBM866'],\
'cp869':['869', 'IBM869', 'CP-GR'],\
'cp874':[],\
'cp875':[],\
'cp932':['932', 'ms932', 'mskanji', 'ms-kanji'],\
'cp949':['949', 'ms949', 'uhc'],\
'cp950':['950', 'ms950'],\
'cp1006':[],\
'cp1026':['ibm1026'],\
'cp1125':['1125', 'ibm1125', 'cp866u', 'ruscii'],\
'cp1140':['ibm1140'],\
'cp1250':['windows-1250'],\
'cp1251':['windows-1251'],\
'cp1252':['windows-1252'],\
'cp1253':['windows-1253'],\
'cp1254':['windows-1254'],\
'cp1255':['windows-1255'],\
'cp1256':['windows-1256'],\
'cp1257':['windows-1257'],\
'cp1258':['windows-1258'],\
'cp65001':[],\
'euc_jp':['eucjp', 'ujis', 'u-jis'],\
'euc_jis_2004':['jisx0213', 'eucjis2004'],\
'euc_jisx0213':['eucjisx0213'],\
'euc_kr':['euckr', 'korean', 'ksc5601', 'ks_c-5601', 'ks_c-5601-1987', 'ksx1001', 'ks_x-1001'],\
'gb18030':['gb18030-2000'],\
'hz':['hzgb', 'hz-gb', 'hz-gb-2312'],\
'iso2022_jp':['csiso2022jp', 'iso2022jp', 'iso-2022-jp'],\
'iso2022_jp_1':['iso2022jp-1', 'iso-2022-jp-1'],\
'iso2022_jp_2':['iso2022jp-2', 'iso-2022-jp-2'],\
'iso2022_jp_2004':['iso2022jp-2004', 'iso-2022-jp-2004'],\
'iso2022_jp_3':['iso2022jp-3', 'iso-2022-jp-3'],\
'iso2022_jp_ext':['iso2022jp-ext', 'iso-2022-jp-ext'],\
'iso2022_jp_kr':['iso2022jp-kr', 'iso-2022-jp-kr'],\
'latin_1':['iso-8859-1', 'iso8859-1', '8859', 'cp819', 'latin', 'latin1', 'L1'],\
'iso8859_2':['iso-8859-2', 'latin2', 'L2'],\
'iso8859_3':['iso-8859-3', 'latin3', 'L3'],\
'iso8859_4':['iso-8859-4', 'latin4', 'L4'],\
'iso8859_5':['iso-8859-5', 'cyrillic'],\
'iso8859_6':['iso-8859-6', 'arabic'],\
'iso8859_7':['iso-8859-7', 'greek', 'greek8'],\
'iso8859_8':['iso-8859-8', 'hebrew'],\
'iso8859_9':['iso-8859-9', 'latin5', 'L5'],\
'iso8859_10':['iso-8859-10', 'latin6', 'L6'],\
'iso8859_11':['iso-8859-11', 'thai'],\
'iso8859_13':['iso-8859-13', 'latin7', 'L7'],\
'iso8859_14':['iso-8859-14', 'latin8', 'L8'],\
'iso8859_15':['iso-8859-15', 'latin9', 'L9'],\
'iso8859_16':['iso-8859-16', 'latin10', 'L10'],\
'johab':['cp1361', 'ms1361'],\
'koi8_r':[],\
'koi8_t':[],\
'koi8_u':[],\
'kz1048':['kz_1048', 'strk1048_2002', 'rk1048'],\
'mac_cyrillic':['maccyrillic'],\
'mac_greek':['macgreek'],\
'mac_iceland':['maciceland'],\
'mac_latin2':['maclatin2', 'maccentraleurope'],\
'mac_roman':['macroman', 'macintosh'],\
'mac_turkish':['macturkish'],\
'ptcp154':['csptcp154', 'pt154', 'cp154', 'cyrillic-asian'],\
'shift_jis':['csshiftjis', 'shiftjis', 'sjis', 's_jis'],\
'shift_jis_2004':['shiftjis2004', 'sjis_2004', 'sjis2004'],\
'shift_jisx0213':['shiftjisx0213', 'sjisx0213', 's_jisx0213'],\
'utf_32':['U32', 'utf32'],\
'utf_32_be':['UTF-32BE'],\
'utf_32_le':['UTF-32LE'],\
'utf_16':['U16', 'utf16'],\
'utf_16_be':['UTF-16BE'],\
'utf_16_le':['UTF-16LE'],\
'utf_7':['U7', 'unicode-1-1-utf-7'],\
'utf_8_sig':[],\
}
global last_enc
last_enc='utf-8'
def s2uni(s):
    global last_enc
    pyv = sys.version_info[0]
    if pyv == 2:
        if isinstance(s, unicode):
            return s
    if pyv == 3:
        if isinstance(s, str):
            return s
    ret = None
    try:
        ret = s.decode(last_enc)
        return ret
    except UnicodeDecodeError:
        try:
            last_enc = chardet.detect(s)["encoding"]
            ret = s.decode(last_enc)
            return ret
        except UnicodeDecodeError:
            for f in CODEC:
                try:
                    ret = s.decode(f)
                except UnicodeDecodeError:
                    continue
                return ret
            raise UnicodeDecodeError

def get_codec(enc):
    enc = enc.lower()
    if (enc.lower() in CODEC) or (enc.upper() in CODEC) or (enc in CODEC):
        return enc
    for e in CODEC:
        if (enc.lower() in CODEC[e]) or (enc.upper() in CODEC[e]) or (enc in CODEC[e]):
            return e
    return None

def econv_sth(o, enc='unicode'):

    if get_codec(enc) == None:
        raise Exception("Not A Valid Format!")
    pyv = sys.version_info[0]
    u = None

    if pyv == 2:
        if not isinstance(o, (unicode, str)):
            return o
        u = s2uni(o)
    elif pyv == 3:
        if not isinstance(o, (str, bytes)):
            return o
        u = s2uni(o)

    if enc in ['unicode', 'uni', 'Unicode']:
        return u
    return u.encode(enc)

def econv_list(l, enc='unicode'):
    if get_codec(enc) == None:
        raise Exception("Not A Valid Format!")
    ret = []
    for o in l:
        ret.append(econv(o, enc))
    return ret

def econv_dict(d, enc='unicode'):
    if get_codec(enc) == None:
        raise Exception("Not A Valid Format!")
    ret = dict()
    for k in d:
        tk = econv(k, enc)
        tv = econv(d[k], enc)
        ret[tk]=tv
    return ret

def econv(o, enc='unicode'):
    if get_codec(enc) == None:
        raise Exception("Not A Valid Format!")
    if isinstance(o, list):
        return econv_list(o, enc)
    if isinstance(o, tuple):
        tmp = list(o)
        tmp = econv_list(tmp, enc)
        return tuple(tmp)
    if isinstance(o, dict):
        return econv_dict(o, enc)
    return econv_sth(o, enc)

def printS(*iterables):
    pyv = sys.version_info[0]
    if pyv == 2:
        import uniout
        for o in iterables:
            o = econv(o, sys.stdout.encoding)
            print(o)
    else:
        for o in iterables:
            o = econv(o)
            print(o)


import json
import codecs
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
