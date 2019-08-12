from __future__ import print_function
from __future__ import division
import importlib
import re
import sys
import time

sys.path.append('alpha-zero-chess/')
from MCTS import MCTS
from chessai.ChessGame import ChessGame as Game
from chessai.tensorflow.NNet import NNetWrapper as NNet
from utils import *
import chess
import numpy as np

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout = Unbuffered(sys.stdout)

import os
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
import argparse

parser = argparse.ArgumentParser(
    description="Specify the config files!")

parser.add_argument(
    "-c",
    dest="config_name",
    type=str,
    default='default',
    help="The preset config file name (under configs/). ")

def main():
    given_args = parser.parse_args()
    args = dotdict(json2load('configs/%s.json'%(given_args.config_name)))
    g = Game()
    nnet = NNet(g, args)
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        
    board = g.getInitBoard()
    mcts = MCTS(g, nnet, args)

    forced = False
    color = 1
    our_time, opp_time = 1000, 1000 # time in centi-seconds
    show_thinking = True

    # print name of chess engine
    print('MemFish')

    stack = []
    while True:
        if stack:
            smove = stack.pop()
        else: smove = input()

        if smove == 'quit':
            break

        elif smove == 'uci':
            print('id name MemFish')
            print('id author zhyack')
            print('uciok')

        elif smove == 'isready':
            print('readyok')

        elif smove == 'ucinewgame':
            stack.append('position fen ' + chess.STARTING_FEN)

        elif smove.startswith('position fen'):
            params = smove.split(' ', 2)
            if params[1] == 'fen':
                fen = params[2]
                board = g.getInitBoard(fen)
                color = 1 if fen.split()[1] == 'w' else -1

        elif smove.startswith('position startpos'):
            params = smove.split(' ')
            #startpos
            board = g.getInitBoard()
            color = 1

            i = 0
            while i < len(params):
                param = params[i]
                if param == 'moves':
                    i += 1
                    while i < len(params):
                        smove = params[i]
                        if color==-1:
                            smove = params[i]
                            smove = smove[0] + chr(7-ord(smove[1])+ord('1')*2) + smove[2] +  chr(7-ord(smove[3])+ord('1')*2) + smove[4:]
                        board, color = g.getNextState(board, color, g.action_dict[smove])
                        i += 1
                i += 1
        elif smove.startswith('usermove'):
            _, smove = smove.split()
            if color==-1:
                smove = smove[0] + chr(7-ord(smove[1])+ord('1')*2) + smove[2] +  chr(7-ord(smove[3])+ord('1')*2) + smove[4:]
            board, color = g.getNextState(board, color, g.action_dict[smove])
            if not forced:
                stack.append('go')
        elif smove.startswith('go'):
            #  default options
            depth = 1000
            movetime = -1

            # parse parameters
            params = smove.split(' ')
            # if len(params) == 1: continue

            i = 0
            while i < len(params):
                param = params[i]
                if param == 'depth':
                    i += 1
                    depth = int(params[i])
                if param == 'movetime':
                    i += 1
                    movetime = int(params[i])
                if param == 'wtime':
                    i += 1
                    our_time = int(params[i])
                if param == 'btime':
                    i += 1
                    opp_time = int(params[i])
                i += 1

            forced = False

            moves_remain = 40

            if our_time<20000:
                mcts.args.numMCTSSims=200
            elif our_time<40000:
                mcts.args.numMCTSSims=300
            elif our_time<80000:
                mcts.args.numMCTSSims=250
            elif our_time<160000:
                mcts.args.numMCTSSims=300
            elif our_time<320000:
                mcts.args.numMCTSSims=350
            else:
                mcts.args.numMCTSSims=400

            start = time.time()
            canonicalBoard = g.getCanonicalForm(board, color)
            valids = g.getValidMoves(canonicalBoard, color)
            smove = g.action_list[np.argmax(mcts.getActionProb(canonicalBoard, temp=0)*valids + valids*0.001)]
            if color==-1:
                smove = smove[0] + chr(7-ord(smove[1])+ord('1')*2) + smove[2] +  chr(7-ord(smove[3])+ord('1')*2) + smove[4:]
            print('bestmove ' + smove)

        elif smove.startswith('time'):
            our_time = int(smove.split()[1])

        elif smove.startswith('otim'):
            opp_time = int(smove.split()[1])

        else:
            pass

if __name__ == '__main__':
    main()
