#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import re
from chess import *
sys.path.append('../../chess-agent/alpha-zero-chess/')
from chessai.ChessGame import ChessGame
from chessai.ChessLogic import *
from pickle import Pickler, Unpickler

piece_map = {'rook':'r', 'pawn':'p', 'knight':'n', 'bishop':'b', 'queen':'q', 'king':'k'}

def getBoardFen(s):
    plist = s.split()
    assert(len(plist)==64)
    fen = ""
    blank_cnt=0
    for j in range(7, -1, -1):
        for i in range(0, 8):
            ps = plist[i*8+j]
            if ps=='eps':
                blank_cnt+=1
                continue
            if blank_cnt>0:
                fen+='%d'%blank_cnt
                blank_cnt=0 
            color, piece = ps.split('_')
            piece = piece_map[piece]
            if color=='white':
                piece = piece.upper()
            fen+=piece
        if blank_cnt>0:
            fen+='%d'%blank_cnt
            blank_cnt=0
        if j>0:
            fen+='/'
    return fen


def getData(raws, texts):
    ret = []
    game = ChessGame()
    board = Board(fen=None)
    for linecnt, game_round in enumerate(raws):
        current_board_str, game_round = game_round.split('<EOC>')
        current_board_str = current_board_str.strip().rstrip()
        previous_board_str, game_round = game_round.split('<EOP>')
        previous_board_str = previous_board_str.strip().rstrip()
        last_move_str, game_round = game_round.split('<EOM>')
        last_move_str = last_move_str.strip().rstrip()
        player = 1
        if last_move_str.startswith('black'):
            player = -1
        previous_board_fen = getBoardFen(previous_board_str)
        # print(board.fen())
        # print(previous_board_fen)
        # print(last_move_str)
        if previous_board_fen!=board.board_fen():
            # print('New Game Here!')
            board = Board(previous_board_fen+' %s KQkq - 0 1'%('w' if player==1 else 'b'))
            game.getInitBoard()
            # print(board.unicode().replace(u'·', u'.'), '\n')
        white_board = board
        if player==-1:
            white_board=board.mirror()
        current_board_fen = getBoardFen(current_board_str)
        canonicalBoard = game.getCanonicalForm(white_board, player)
        valids = game.getValidMoves(white_board, player, mode='list')
        last_move = None
        # print([move.uci() for move in board.legal_moves])
        for move in board.legal_moves:
            board.push(move)
            if board.board_fen() == current_board_fen:
                last_move = move
                break
            board.pop()
        if last_move==None:
            # print(board.unicode().replace(u'·', u'.'), '\n')
            # print(BaseBoard(current_board_fen).unicode().replace(u'·', u'.'), '\n')
            # print('Find Invalid moves @ %d'%(linecnt))
            # ret.append(None)
            continue
        # print(board.unicode().replace(u'·', u'.'))
        actual_move = last_move.uci()
        if player == -1:
            actual_move = actual_move[0] + chr(7-ord(actual_move[1])+ord('1')*2) + actual_move[2] +  chr(7-ord(actual_move[3])+ord('1')*2) + actual_move[4:]
        actual_move = game.action_dict[actual_move]
        ret.append((canonicalBoard, [actual_move], 0.0, valids, texts[linecnt]))
    return ret



flist = os.listdir('../crawler/saved_files/')

for fname in flist:
    if not fname.endswith('.single.che'):
        continue
    f = codecs.open('../crawler/saved_files/'+fname, 'r', 'UTF-8')
    raw_lines = f.readlines()
    f.close()
    f = codecs.open('../crawler/saved_files/'+fname[:-4]+'.en', 'r', 'UTF-8')
    text_lines = f.readlines()
    f.close()
    data_lines = getData(raw_lines, text_lines)
    print(fname[:-11]+'.pickle', len(data_lines))
    with open(fname[:-11]+'.pickle', "wb+") as f:
        Pickler(f).dump(data_lines)
    f.closed

f = codecs.open('../crawler/saved_files/train.che-eng.single.en', 'r', 'UTF-8')
d = {}
for line in f.readlines():
    for word in line.split():
        if word not in d:
            d[word]=0
        d[word]+=1
f.close()
f = codecs.open('text.dict', 'w', 'UTF-8')
for w in d.keys():
    if d[w]>2:
        f.write(w+'\n')
f.close()


