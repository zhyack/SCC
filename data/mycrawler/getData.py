#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from html.parser import HTMLParser
import re
import nltk
from nltk.tokenize import WordPunctTokenizer
sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

class MyHTMLParser(HTMLParser):
    def __init__(self, bias=0, is_print=False):
        self.next_link = None
        self.pgns = []
        self.comments = []
        self.table_depth = -10000
        self.tr_depth = 0
        self.td_depth = 0
        self.tr_cnt = 0
        self.td_cnt = 0
        self.sub_comment_ready = False
        self.sub_comment_near = False
        self.now_i_am_in = 'body'
        self.pgn_line = (2+bias)%4
        self.comment_line = (3+bias)%4
        self.is_print = is_print
        HTMLParser.__init__(self)
        
    def handle_starttag(self, tag, attrs):
        attr_dict = {}
        for t in attrs:
            if len(t)==2:
                attr_dict[t[0]]=t[1]
        if tag=='table':
            if 'class' in attr_dict and attr_dict['class']=='dialog':
                self.table_depth=0
            self.table_depth += 1
        elif tag=='tr' and self.table_depth==1:
            self.tr_depth += 1
            if self.tr_depth == 1:
                self.tr_cnt += 1
                self.td_cnt = 0
        elif tag=='td' and self.table_depth==1:
            self.td_depth += 1
            if self.td_depth == 1:
                self.td_cnt += 1
        elif tag=='a' and self.table_depth==2 and 'title' in attr_dict and attr_dict['title']=='next page':
            self.next_link = attr_dict['href']
        elif tag=='div' and self.table_depth==1 and 'class' in attr_dict and attr_dict['class']=='hlt_text':
            self.sub_comment_near = True

        self.now_i_am_in = tag


    def handle_endtag(self, tag):
        if tag=='table':
            self.table_depth -= 1
            if self.table_depth==0:
                self.table_depth=-10000
        elif tag=='tr' and self.table_depth==1:
            self.tr_depth -= 1
        elif tag=='td' and self.table_depth==1:
            self.td_depth -= 1
        elif tag=='div' and self.sub_comment_near:
            self.sub_comment_near=False
            self.sub_comment_ready=True



    def handle_startendtag(self, tag, attrs):
        pass

    def handle_data(self, data):
        if self.is_print:
            print(self.table_depth, self.tr_depth, self.tr_cnt, self.td_depth, self.td_cnt, data)
        if self.table_depth==1:
            data = data.replace('\n', ' ').replace('\r', ' ').replace('\xa0', ' ').strip().rstrip()
            data = re.sub(' +', ' ', data)
            if self.tr_depth==1:
                if self.tr_cnt%4==self.pgn_line and self.now_i_am_in=='td':
                    if self.td_depth==1:
                        if self.td_cnt==1:
                            # print('pgn@%d'%(self.tr_cnt), data)
                            self.pgns.append(data)
                            self.comments.append([])
                        elif self.td_cnt==2:
                            data = sen_tokenizer.tokenize(data)
                            # print('comments@%d'%(self.tr_cnt), data)
                            self.comments[-1]+=data
                            
                elif self.tr_cnt%4==self.comment_line and self.sub_comment_ready:
                    if self.td_depth==1:
                        if self.td_cnt == 1:
                            # print('sub_comments@%d'%(self.tr_cnt), data)
                            # self.comments[-1]+=[data]
                            self.sub_comment_ready = False
    

    def handle_comment(self, data):
        pass

    def handle_entityref(self, name):
        pass

    def handle_charref(self, name):
        pass


completed_log = "data/completed.json"

category = ['train', 'valid', 'test']

from crawler_utils import *
socket.setdefaulttimeout(5)
from data_utils import *
import chess
import time
import io
import os
from nltk.tokenize import word_tokenize

completed = json2load(completed_log)

for cat in category:
    if cat not in completed:
        completed[cat] = {}
    flinks = codecs.open('data/%s_links.p'%cat, 'r', 'UTF-8')
    lines = flinks.readlines()
    flinks.close()
    link_cnt = 0
    
    fcat_games = codecs.open('data/%s.games'%cat, 'w', 'UTF-8')
    fcat_comments = codecs.open('data/%s.comments'%cat, 'w', 'UTF-8')

    for line in lines:
        if line.find('http')==-1:
            continue
        link = line[line.find('http'):]
        game_pgn = []
        game_comment = []
        link_inner_cnt = 0

        if link in completed[cat]:
            print('Reading %s...'%(link))
            for i in range(100):
                content = None
                if os.path.exists('data/raw_data/%s-%d-%d.html'%(cat, link_cnt, i)):
                    print('data/raw_data/%s-%d-%d.html'%(cat, link_cnt, i))
                    f = codecs.open('data/raw_data/%s-%d-%d.html'%(cat, link_cnt, i), 'r', 'UTF-8')
                    content = '\n'.join(f.readlines())
                    f.close()
                else:
                    break
                for try_bias in range(5):
                    if try_bias==4:
                        # content_parser = MyHTMLParser(bias=1, is_print=True)
                        # content_parser.feed(content)
                        # raise Exception("It cannot be parsed... %s"%link)
                        del(completed[cat][link])
                        break
                    content_parser = MyHTMLParser(bias=try_bias)
                    try:
                        content_parser.feed(content)
                    except IndexError:
                        continue
                    pgns, comments = content_parser.pgns, content_parser.comments
                    if len(pgns)==0:
                        print('%d-%d get nothing??? retry...'%(i,try_bias))
                        continue
                    if '' in pgns:
                        print('%d-%d blank pgn??? retry...'%(i,try_bias))
                        continue
                    game_pgn.extend(pgns)
                    game_comment.extend(comments)
                    break
        if link not in completed[cat]:
            game_pgn = []
            game_comment = []
            print('Getting %s...'%(link))
            try_bias = 5
            while(True):
                if try_bias==0:
                    break
                link_path = 'data/raw_data/%s-%d-%d.html'%(cat, link_cnt, link_inner_cnt)
                _, content = getPage(link, link_path)
                if content==None or content.find('ANNOTATED GAME')==-1:
                    print('%s cannot find symbol ANNOTATED GAME, retry...'%link)
                    continue
                content_parser = MyHTMLParser(bias=try_bias)
                try:
                    content_parser.feed(content)
                except IndexError:
                    pass
                pgns, comments = content_parser.pgns, content_parser.comments
                if len(pgns)==0:
                    print('%d-%d get nothing??? retry...'%(link_inner_cnt,try_bias))
                    try_bias -= 1
                    continue
                if '' in pgns:
                    print('%d-%d blank pgn??? retry...'%(link_inner_cnt,try_bias))
                    try_bias -= 1
                    continue
                game_pgn.extend(pgns)
                game_comment.extend(comments)
                if content_parser.next_link:
                    link = 'https://gameknot.com'+content_parser.next_link
                    link_inner_cnt+=1
                    try_bias = 5
                    print('Have next page... @%d'%(link_inner_cnt))
                else:
                    break
            if try_bias==0:
                print('Missing something')
                continue

        print('Got! - %d'%(link_cnt))
        n_data = len(game_pgn)
        pgn_fornow = ''
        # print(game_comment)
        fgames = codecs.open('data/raw_data/%s-%d.games'%(cat, link_cnt), 'w', 'UTF-8')
        fcomments = codecs.open('data/raw_data/%s-%d.comments'%(cat, link_cnt), 'w', 'UTF-8')
        for i in range(n_data):
            if game_pgn[i].find('...')!=-1:
                pgn_fornow += ' ' + game_pgn[i][game_pgn[i].find('...')+3:].strip()
            else:
                pgn_fornow += ' ' + game_pgn[i].strip()
            
            pgn_list = pgn_fornow.replace('e.p.+', '').replace('e.p.', '').split()
            for t in range(len(pgn_list)//3):
                w = pgn_list[t*3][:-1]
                try:
                    dw = int(w)
                except Exception:
                    print(game_pgn)
                    raise Exception('error pgn in %s : %s'%(line[line.find('http'):], pgn_fornow))
                if dw != t+1:
                    print(game_pgn)
                    raise Exception('error pgn in %s : %s'%(line[line.find('http'):], pgn_fornow))           
            # pgn = io.StringIO(pgn_fornow+' *')
            
            for comment in game_comment[i]:
                if len(comment)==0:
                    continue
                comment=' '.join(word_tokenize(comment))
                fcat_games.write(pgn_fornow+' *\n')
                fcat_comments.write(comment+'\n')
                fgames.write(pgn_fornow+' *\n')
                fcomments.write(comment+'\n')
        fgames.close()
        fcomments.close()
        fgames = codecs.open('data/raw_data/%s-%d.games'%(cat, link_cnt), 'r', 'UTF-8')
        fcomments = codecs.open('data/raw_data/%s-%d.comments'%(cat, link_cnt), 'r', 'UTF-8')
        if len(fgames.readlines())!=len(fcomments.readlines()):
            fgames.close()
            fcomments.close()
            raise Exception('data/raw_data/%s-%d.games |||| Not Equal!'%(cat, link_cnt))
        fgames.close()
        fcomments.close()
        print('Saved!')
        completed[cat][line[line.find('http'):]]=True
        save2json(completed, completed_log)
        link_cnt += 1
    fcat_games.close()
    fcat_comments.close()

            



            


