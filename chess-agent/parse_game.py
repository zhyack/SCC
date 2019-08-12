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

ratios = [0.00, 0.08, 0.76, 2.66, 5.88, 10.42, 16.69, 23.31, 30.61, 38.42, 45.86, 52.86, 59.21, 65.37, 70.45, 74.92, 78.84, 82.17, 84.92, 87.37, 89.17, 90.90, 92.28, 93.50, 94.41, 95.20, 95.86, 96.32, 96.64, 96.99, 97.29, 97.51, 97.65, 97.82, 97.93, 98.04, 98.15, 98.22, 98.23, 98.34, 98.35, 98.39, 98.41, 98.43, 98.44, 98.49, 98.51, 98.61, 98.80, 98.87, 98.91, 98.92, 98.93, 98.93, 98.92, 98.97, 99.03, 99.01, 99.00, 99.02, 99.01, 99.06, 99.08, 99.09, 99.08, 99.08, 99.09, 99.13, 99.13, 99.19, 99.18, 99.21, 99.24, 99.24, 99.24, 99.25, 99.27, 99.30, 99.30, 99.29, 99.30, 99.38, 99.41, 99.43, 99.49, 99.47, 99.49, 99.49, 99.52, 99.52, 99.55, 99.55, 99.69, 99.68, 99.79, 99.79, 99.78, 99.78, 99.77, 99.78, 99.81, 99.76, 99.77, 99.74, 99.72, 99.86, 99.85, 99.89, 99.92, 99.90, 99.90, 99.96, 99.94, 99.93, 99.91, 99.93, 99.90, 99.87, 99.87, 99.92, 99.89, 99.92, 99.91, 99.88, 99.82, 99.78, 99.74, 99.70, 99.73, 99.65, 99.67, 99.78, 99.77, 99.60, 99.59, 99.75, 99.74, 99.37, 99.36, 99.53, 99.51, 99.59, 99.63, 99.62, 99.83, 99.71, 99.58, 99.57, 99.94, 99.87, 99.87, 99.72, 99.37, 99.49, 99.70, 99.53, 99.52, 99.50, 99.49, 99.74, 99.64, 99.54, 99.52, 99.51, 99.70, 99.69, 99.79, 99.67, 99.66, 99.89, 99.76, 99.52, 99.51, 99.87, 99.87, 100.00, 100.00, 99.86, 99.71, 99.85, 100.00, 100.00, 100.00, 99.84, 99.83, 99.31, 99.47, 99.10, 99.07, 99.05, 99.22, 99.21, 98.78, 98.53, 98.93, 99.35, 99.33, 99.31, 99.76, 99.52]

def read_games(fn):
    f = open(fn)

    while True:
        try:
            print('fetchig a game from %s...'%(fn))
            g = chess.pgn.read_game(f)
        except KeyboardInterrupt:
            raise
        except:
            continue

        if not g:
            break

        # print('fetched a game from %s!'%(fn))
        yield g


def initUCI():
    handler = chess.uci.InfoHandler()
    engine = chess.uci.popen_engine('../arena/Engines/Stockfish/stockfish_8_x64')
    engine.info_handlers.append(handler)
    return engine, handler

# def getScore(engine, handler, board, depth=15):
#     left_try = 2
#     while True:
#         try:
#             engine.position(board)
#             evaluation = engine.go(depth=depth)
#             ret = handler.info["score"][1].cp/100.0
#             return ret
#         except TypeError:
#             if left_try==0:
#                 return None
#             left_try-=1

def getScore(player, winner):
    return player*winner

def parse_game(g):
    # Generate all boards
    game = ChessGame.ChessGame()
    board = game.getInitBoard()
    gn = g.end()
    gns = []
    while gn:
        gns.append(gn)
        gn = gn.parent

    n = len(gns)-1

    scores = []
    actions = []
    actions_masks = []
    boards = []
    current_player = 1
    replays = []
    if int(g.headers['WhiteElo'])<2000 or int(g.headers['BlackElo'])<2000:
        return replays
    result = g.headers['Result']
    if result == '0-1':
        winner = -1
    elif result == '1-0':
        winner = 1
    elif random.random()<0.5:
        winner = 0.05
    else:
        winner = -0.05
    # print(result, winner)
    for i in range(n-1, -1, -1):

        actual_move = gns[i].move.uci()
        if current_player == -1:
            actual_move = actual_move[0] + chr(7-ord(actual_move[1])+ord('1')*2) + actual_move[2] +  chr(7-ord(actual_move[3])+ord('1')*2) + actual_move[4:]
        actual_move = game.action_dict[actual_move]

        step = n-1-i
        r = 1.0
        if step<len(ratios):
            r = min(1.0, ratios[step]/100.0+0.00001)
        if random.random()>r:
            board, current_player = game.getNextState(board, current_player, actual_move)
            continue

        score = getScore(current_player, winner)

        canonicalBoard = game.getCanonicalForm(board, current_player)

        valids = game.getValidMoves(board, 1, mode='list')

        replays.append((canonicalBoard, [actual_move], score, valids, None))
        # print(board.unicode().replace(u'·', u'.'))
        # print(game.action_list[actual_move])
        # print([game.action_list[m] for m in valids])



        board, current_player = game.getNextState(board, current_player, actual_move)

    return replays


def saveTrainExamples(trainExamples, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, filename), "wb+") as f:
        Pickler(f).dump(trainExamples)
    f.closed

def loadTrainExamples(examplesFile):
    if not os.path.isfile(examplesFile):
        raise Exception("File with trainExamples not found. %s"%(examplesFile))
    else:
        print("File with trainExamples found. Read it. %s"%(examplesFile))
        with open(examplesFile, "rb") as f:
            ExamplesHistory = Unpickler(f).load()
        f.closed
        return ExamplesHistory

def read_all_games(fn_in, fn_out):
    # fn_in, fn_out = fns
    # engine, handler = initUCI()
    # size = 0
    game_cnt = 0
    cnt = 0
    all_replays = []
    for game in read_games(fn_in):
        game_cnt += 1
        print('parsing the %d-th game now...'%(game_cnt))
        # replays = parse_game(game, engine, handler)
        replays = parse_game(game)
        # print('parsed!')
        if len(replays)==0:
            continue

        n = len(replays)

        cnt += n
        all_replays.extend(replays)
        print('here we got %d replays~'%(cnt))
        if (game_cnt%5000 == 0):
            print('saving...')
            saveTrainExamples(all_replays, 'replays', fn_out)
            print('saved!')
    saveTrainExamples(all_replays, 'replays', fn_out)

def read_all_games_2(a):
    return read_all_games(*a)

def parse_dir():
    files = []
    d = '../deep-pink/games/'
    dout = 'replays/'
    for fn_in in os.listdir(d):
        if not fn_in.endswith('.pgn'):
            continue
        fn_out = fn_in.replace('.pgn', '.pickle')
        if fn_out in os.listdir(dout):
            continue
        fn_in = os.path.join(d, fn_in)
        files.append((fn_in, fn_out))

    # pool = multiprocessing.Pool(processes=10)
    # pool.map(read_all_games_2, files)

    for fns in files:
        read_all_games(fns[0], fns[1])


if __name__ == '__main__':
    parse_dir()
