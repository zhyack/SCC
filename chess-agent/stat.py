#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import chess, chess.pgn, chess.uci
import numpy
import sys
import os
import multiprocessing
import itertools
import random
from pickle import Pickler, Unpickler
import numpy as np

sys.path.append('alpha-zero-chess/')

import chessai.ChessGame as ChessGame
import chessai.ChessLogic as ChessLogic

global board_cnt, board_step_cnt, step_cnt

import math


def read_games(fn):
    f = open(fn)

    while True:
        try:
            g = chess.pgn.read_game(f)
        except KeyboardInterrupt:
            raise
        except:
            continue

        if not g:
            break

        # print('fetched a game from %s!'%(fn))
        yield g



def getScore(player, winner):
    return player*winner

def parse_game(g):
    global board_cnt, step_cnt, board_step_cnt
    # Generate all boards
    game = ChessGame.ChessGame()
    board = game.getInitBoard()
    gn = g.end()
    gns = []
    while gn:
        gns.append(gn)
        gn = gn.parent

    n = len(gns)-1

    current_player = 1

    for i in range(n-1, -1, -1):

        actual_move = gns[i].move.uci()
        if current_player == -1:
            actual_move = actual_move[0] + chr(7-ord(actual_move[1])+ord('1')*2) + actual_move[2] +  chr(7-ord(actual_move[3])+ord('1')*2) + actual_move[4:]
        actual_move = game.action_dict[actual_move]

        fen = board.fen()
        if fen not in board_cnt:
            board_cnt[fen] = 0
            board_step_cnt[fen] = set()
        board_cnt[fen]+=1
        step = n-1-i
        if step < 200:
            if fen not in step_cnt[step]:
                step_cnt[step][fen] = 0
            step_cnt[step][fen] += 1
            board_step_cnt[fen].add(step)


        board, current_player = game.getNextState(board, current_player, actual_move)


def read_all_games(fn_in):
    # fn_in, fn_out = fns
    # engine, handler = initUCI()
    # size = 0
    game_cnt = 0
    print('dealing with %s'%(fn_in))
    for game in read_games(fn_in):
        game_cnt += 1
        parse_game(game)
        if (game_cnt%1000 == 0):
            break

def tell_me():
    global board_cnt, step_cnt, board_step_cnt
    different_boards = [0]*200
    for i in range(200):
        for s in step_cnt[i]:
            if board_cnt[s]-step_cnt[i][s]==0:
                different_boards[i]+=1
    actual_boards = [0]*200
    for i in range(200):
        for s in step_cnt[i]:
            actual_boards[i]+=step_cnt[i][s]
    print('Different Boards: ', different_boards)
    print('Stat Boards:', actual_boards)
    def trydiv(a,b):
        if b==0:
            return 'NAN'
        else:
            return '%.2f%%'%(a/b*100)
    print('Ratios:', [trydiv(different_boards[i], actual_boards[i]) for i in range(200)])
    tfidf = [0.0]*200
    N = 200
    for i in range(200):
        for b in step_cnt[i]:
            tf = step_cnt[i][b]
            df = len(board_step_cnt[b])
            tfidf[i] += math.log10(tf+1)*math.log10(N/df)
        tfidf[i] = int(tfidf[i])
    print('TF-IDF:', tfidf)
    r = 0.0
    s = sum(actual_boards)
    for i in range(200):
        r += min(different_boards[i]/actual_boards[i]+0.0001,1.0) *actual_boards[i]/s
    print(r)

def parse_dir():
    files = []
    d = '../deep-pink/games/'
    for fn_in in os.listdir(d):
        if not fn_in.endswith('.pgn'):
            continue
        files.append(d+fn_in)

    for fn in files:
        read_all_games(fn)
        tell_me()


if __name__ == '__main__':
    global board_cnt, step_cnt, board_step_cnt
    step_cnt = []
    for _ in range(200):
        step_cnt.append({})
    board_cnt = {}
    board_step_cnt = {}
    parse_dir()
