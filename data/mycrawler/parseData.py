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
import io

sys.path.append('../../chess-agent/alpha-zero-chess/')

import chessai.ChessGame as ChessGame
import chessai.ChessLogic as ChessLogic




def saveTrainExamples(trainExamples, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, filename), "wb+") as f:
        Pickler(f).dump(trainExamples)
    f.closed

def getReplay(pgn_str, text, clss):
    game = ChessGame.ChessGame()
    board = game.getInitBoard()
    pgn = io.StringIO(pgn_str)
    g = chess.pgn.read_game(pgn)
    gn = g.end()
    gns = []
    while gn:
        gns.append(gn)
        gn = gn.parent
    n = len(gns)-1
    winner = 0.05
    score = None
    action = None
    action_mask = None
    canonicalBoard = None
    current_player = 1
    replay = None
    actual_move = None
    valid = None
    for i in range(n-1, -1, -1):
        actual_move = gns[i].move.uci()
        if current_player == -1:
            actual_move = actual_move[0] + chr(7-ord(actual_move[1])+ord('1')*2) + actual_move[2] +  chr(7-ord(actual_move[3])+ord('1')*2) + actual_move[4:]
        actual_move = game.action_dict[actual_move]

        score = current_player*winner

        canonicalBoard = game.getCanonicalForm(board, current_player)

        valid = game.getValidMoves(board, 1, mode='list')

        board, current_player = game.getNextState(board, current_player, actual_move)

    replay = (canonicalBoard, [actual_move], score, valid, '@\t@'.join([' '.join(clss), text]))
    return replay

from data_utils import *
completed_log = "data/completed.json"
category = ['train', 'valid', 'test']

completed = json2load(completed_log)

for cat in category:
    if not (os.path.exists('data/%s.games'%cat) and os.path.exists('data/%s.comments'%cat)):
        continue
    fcat_games = codecs.open('data/%s.games'%cat, 'r', 'UTF-8')
    fcat_comments = codecs.open('data/%s.comments'%cat, 'r', 'UTF-8')
    games = fcat_games.readlines()
    comments = fcat_comments.readlines()
    fcat_games.close()
    fcat_comments.close()
    assert(len(games)==len(comments))
    classes = [[] for _ in range(len(games))]
    pred_labels_mapping = [1, 2, 4, 5]
    for k in range(4):
        fcat_classes = codecs.open('data/%s.comments.pred_labels_%d'%(cat, k), 'r', 'UTF-8')
        cc = pred_labels_mapping[k]
        for i, line in enumerate(fcat_classes):
            if line[0]=='1':
                classes[i].append(str(cc))
        fcat_classes.close()
    fcat_classes = codecs.open('data/%s.comments.pseudoLabels'%(cat), 'r', 'UTF-8')
    for i, line in enumerate(fcat_classes):
        # print(line)
        pp = eval(line.split('||||')[1])[1]
        if pp>0:
            classes[i].append(str(3))
    fcat_classes.close()
    replays = []
    for i, g, c, clss in zip(range(len(games)), games, comments, classes):
        print('%s-[%d/%d]...'%(cat, i, len(games)))
        replay = getReplay(g, c, clss)
        replays.append(replay)
    saveTrainExamples(replays, 'data/', '%s.pickle'%cat)
