import chess, chess.pgn, chess.uci
import numpy
import sys
import os
import multiprocessing
import itertools
import random
import h5py


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

        print('fetched a game from %s!'%(fn))
        yield g


def bb2array(b, flip=False):
    x = numpy.zeros(64, dtype=numpy.int8)

    for pos, piece in enumerate(b.piece_map()):
        if piece != 0:
            color = int(bool(b.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))
            col = int(pos % 8)
            row = int(pos / 8)
            if flip:
                row = 7-row
                color = 1 - color

            piece = color*7 + piece

            x[row * 8 + col] = piece

    return x

def initUCI():
    handler = chess.uci.InfoHandler()
    engine = chess.uci.popen_engine('../arena/Engines/Stockfish/stockfish_8_x64')
    engine.info_handlers.append(handler)
    return engine, handler


def getScore(engine, handler, board, depth=15):
    left_try = 2
    while True:
        try:
            engine.position(board)
            evaluation = engine.go(depth=depth)
            ret = handler.info["score"][1].cp/100.0
            return ret
        except TypeError:
            if left_try==0:
                return None
            left_try-=1


def parse_game(g, engine, handler):
    # Generate all boards
    gn = g.end()
    # if not gn.board().is_game_over():
    #     return None
    gns = []
    while gn:
        gns.append(gn)
        gn = gn.parent

    n = len(gns)-1

    scores = []
    scores_borad = []
    boards_cur = []
    boards_prev = []
    actual_move = []
    legal_moves = []
    cached = None
    for i in range(n):
        gn_cur = gns[i]
        gn_prev = gns[i+1]
        if not cached:
            sc_cur = getScore(engine, handler, gn_cur.board(), min(23, int((n-i)/n*7.0)+16))
        else:
            sc_cur = cached
        sc_prev = getScore(engine, handler, gn_prev.board(), min(23, int((n-i)/n*7.0)+16))
        cached = sc_prev
        if sc_cur==None or sc_prev==None:
            # print('skip')
            continue
        # print(gn_cur.move, sc_cur, sc_prev, -sc_cur-sc_prev)
        scores.append(-sc_cur-sc_prev)
        boards_cur.append(bb2array(gn_cur.board(), gn_cur.board().turn))
        actual_move.append(str(gn_cur.move))
        boards_prev.append(bb2array(gn_prev.board(), gn_prev.board().turn))
        legal_moves.append([str(m) for m in gn_prev.board().legal_moves])

    # replays = []
    n = len(scores)
    # for t in range(n-1):
    #     replays.append(boards[t+1], actual_move[t], scores[t+1]+scores[t], boards[t], legal_moves[t+1])

    return (boards_prev, actual_move, scores, boards_cur, legal_moves)


def read_all_games(fn_in, fn_out):
    engine, handler = initUCI()
    if sys.version_info[0] == 2:
        dst = h5py.special_dtype(vlen=unicode)
    else:
        dst = h5py.special_dtype(vlen=str)
    # (fn_in, fn_out) = fns
    g = h5py.File(fn_out, 'w')
    sp, s = [g.create_dataset(d, (0, 64), dtype='b', maxshape=(None, 64), chunks=True) for d in ['sp', 's']]
    A = g.create_dataset('A', (0, 0), dtype=dst, maxshape=(None, None))
    a = g.create_dataset('a', (0,), dtype=dst, maxshape=(None,))
    r = g.create_dataset('r', (0,), dtype='b', maxshape=(None,))
    size = 0
    cnt = 0
    for game in read_games(fn_in):
        print('parsing the game now...')
        replays = parse_game(game, engine, handler)
        print('parsed!')
        if len(replays)==0:
            continue
        prev_board, action, reward, cur_borad, lacs = replays

        n = len(action)

        while cnt + n >= size:
            g.flush()
            size = 2 * size + 1
            print 'resizing to', size
            [d.resize(size=size, axis=0) for d in (sp, s, A, a, r)]

        for i in range(cnt, cnt+n):
            sp[i] = prev_board[i-cnt]
            s[i] = cur_borad[i-cnt]
            A[i] = lacs[i-cnt]
            a[i] = action[i-cnt]
            r[i] = reward[i-cnt]

        cnt += n
        print('here we got %d replays~'%(cnt))

    [d.resize(size=cnt, axis=0) for d in (sp, s, A, a, r)]
    g.close()

def read_all_games_2(a):
    return read_all_games(*a)

def parse_dir():
    files = []
    # d = '../deep-pink/games/'
    d = 'replays/'
    for fn_in in os.listdir(d):
        if not fn_in.endswith('.pgn'):
            continue
        fn_out = 'replays/'+fn_in.replace('.pgn', '.hdf5')
        fn_in = os.path.join(d, fn_in)
        # if not os.path.exists(fn_out):
        #     files.append((fn_in, fn_out))
        files.append((fn_in, fn_out))

    # pool = multiprocessing.Pool(processes=5)
    # pool.map(read_all_games_2, files)

    for fns in files:
        read_all_games(fns[0], fns[1])

if __name__ == '__main__':
    parse_dir()
