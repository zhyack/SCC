#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import os
sys.path.append('../')
from Game import Game
from .ChessLogic import *
import numpy as np

class ChessGame(Game):
    def __init__(self, buildDict=False):
        self.action_history = []
        # if not os.path.isfile('/home/zanghy/ICC/chess-agent/alpha-zero-chess/chessai/actions.dict'):
        if buildDict:
            buildActionDict('/home/zanghy/ICC/chess-agent/alpha-zero-chess/chessai/actions.dict')
        self.action_list, self.action_dict = getActionDict('/home/zanghy/ICC/chess-agent/alpha-zero-chess/chessai/actions.dict')
        if buildDict:
            print('Get %d actions!'%(len(self.action_list)))

    def getInitBoard(self, fen=chess.STARTING_FEN):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        self.action_history = ['?a', '?b', '?c', '?d', '?e', '?f']
        return chess.Board(fen)

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (8,8)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return len(self.action_list)

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        board = self.restoreOriginBoard(board)
        if not isinstance(action, chess.Move):
            action = chess.Move.from_uci(self.action_list[action])
        if action not in board.legal_moves:
            # print(board.unicode().replace(u'·', u'.'))
            raise Exception("Move %s is not a legal move in board %s."%(action.uci(), board.fen()))
        board.push(action)
        new_player = -player
        self.action_history.append(action.uci())
        while(len(self.action_history)>6):
            del(self.action_history[0])
        return (board.mirror(), new_player)

    def isValidMove(self, board, move):
        moves = [self.action_dict[m.uci()] for m in board.legal_moves]
        return move in moves

    def getValidMoves(self, canonicalBoard, player, mode='vector'):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        board = self.restoreOriginBoard(canonicalBoard)
        moves = [self.action_dict[m.uci()] for m in board.legal_moves]
        if mode=='list':
            return moves
        valids = [0]*self.getActionSize()
        if len(moves)==0:
            raise Exception("There is no legal move here! %s"%(board.fen()))
        for m in moves:
            valids[m] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        board = self.restoreOriginBoard(board)
        ret = 0
        result = board.result()
        mapping = {'*':0, '1-0':1*player, '0-1':-1*player, '1/2-1/2':0.1}

        return mapping[result]

    def restoreOriginBoard(self, canonicalBoard):
        new_board = canonicalBoard
        if not isinstance(canonicalBoard, chess.Board):
            board, rep1, rep2, mcnt, npcnt, cast11, cast12, cast21, cast22, _ = canonicalBoard
            new_board = chess.Board(board)
            new_board.fullmove_number=mcnt
            new_board.halfmove_clock=npcnt
            if cast11:
                new_board.castling_rights |= chess.BB_A1
            if cast12:
                new_board.castling_rights |= chess.BB_A8
            if cast21:
                new_board.castling_rights |= chess.BB_H1
            if cast22:
                new_board.castling_rights |= chess.BB_H8

        return new_board

    def getCanonicalForm(self, originBoard, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        board = originBoard
        canonical_board = [board.fen()]
        i = -1
        rep = 1
        last_move = self.action_history[i]
        for j in range(1,3):
            if self.action_history[i-j*2-1]:
                rep+=1
            else:
                break
        canonical_board.append(rep)
        rep = 1
        last_move = self.action_history[i-1]
        for j in range(1,3):
            if self.action_history[i-j*2-1]:
                rep+=1
            else:
                break
        canonical_board.append(rep)
        canonical_board.append(originBoard.fullmove_number)
        canonical_board.append(originBoard.halfmove_clock)
        if bool(originBoard.castling_rights & chess.BB_A1):
            canonical_board.append(1)
        else:
            canonical_board.append(0)
        if bool(originBoard.castling_rights & chess.BB_A8):
            canonical_board.append(1)
        else:
            canonical_board.append(0)
        if bool(originBoard.castling_rights & chess.BB_H1):
            canonical_board.append(1)
        else:
            canonical_board.append(0)
        if bool(originBoard.castling_rights & chess.BB_H8):
            canonical_board.append(1)
        else:
            canonical_board.append(0)
        canonical_board.append(player)
        return canonical_board


    def getSymmetries(self, canonicalBoard, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """


        return([(canonicalBoard, pi)])

    def stringRepresentation(self, canonicalBoard):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        board, rep1, rep2, mcnt, npcnt, cast11, cast12, cast21, cast22, _ = canonicalBoard
        return board

    def getScore(self, board, player):
        pass

def display(board):
    pass
