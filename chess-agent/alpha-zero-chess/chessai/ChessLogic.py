import chess
import numpy
import codecs

dirs = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
knight_dirs = [(1,2), (-1,2), (-2,1), (-2,-1), (-1,-2), (1,-2), (2,-1), (2,1)]
promotion_dirs = [(1,1), (-1,1), (0,1)]
promotion_choice = ['q', 'n', 'r', 'b']
asize = 76


def buildActionDict(pf='chessai/actions.dict'):
    def mkstr(a,b,c,d,s=""):
        return chr(ord('a')+a)+chr(ord('1')+b) + chr(ord('a')+c)+chr(ord('1')+d)+s
    book = []
    for i in range(8):
        st_x = i
        for j in range(8):
            st_y = j
            for d in dirs:
                for length in range(1,8):
                    en_x = st_x+d[0]*length
                    en_y = st_y+d[1]*length
                    if (en_x>=0 and en_x<8 and en_y>=0 and en_y<8):
                        book.append(mkstr(st_x, st_y, en_x, en_y, ''))
            for d in knight_dirs:
                en_x = st_x+d[0]
                en_y = st_y+d[1]
                if (en_x>=0 and en_x<8 and en_y>=0 and en_y<8):
                    book.append(mkstr(st_x, st_y, en_x, en_y, ''))
            if st_y == 6:
                for d in promotion_dirs:
                    for choice in promotion_choice:
                        en_x = st_x+d[0]
                        en_y = st_y+d[1]
                        if (en_x>=0 and en_x<8):
                            book.append(mkstr(st_x, st_y, en_x, en_y, choice))
            # book.append(mkstr(st_x, st_y, st_x, st_y))
    f = codecs.open(pf, 'w', 'UTF-8')
    f.write('\n'.join(book))
    f.close()
    return book

def getActionDict(pf='chessai/actions.dict'):
    f = codecs.open(pf, 'r', 'UTF-8')
    book = f.readlines()
    book = [a.strip() for a in book]
    f.close()
    d = {}
    for i,a in enumerate(book):
        d[a] = i
    return book, d

def getRealMove(d, m):
    return d[m]

def getMappedMove(rd, m):
    return rd[m]

max_moves = 100
max_no_progress = 100
fsize = 20

def getBoardFeatures(canonicalBoard):
    def getPieces(board, dtype=numpy.float32):
        x = numpy.zeros([8,8,12], dtype=dtype)
        mapping = {'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5, 'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}
        pieces = board.piece_map()
        for pos in pieces:
            piece = pieces[pos]
            col = int(pos % 8)
            row = int(pos / 8)
            symbol = mapping[piece.symbol()]
            x[row][col][symbol] = 1
        return x
    def getThreats(board, dtype=numpy.float32):
        x = numpy.zeros([8,8,64], dtype=dtype)
        for i in range(64):
            atts = board.attackers(chess.BLACK, i)
            for j in range(64):
                if j in atts:
                    x[j//8][j%8][i]=1
        return x
    def getValuePlane(v, dtype=numpy.float32):
        x = numpy.full([8,8,1], v, dtype=dtype)
        return x
    def getNorm(v, min_bound=0.0, max_bound=1.0):
        assert(v>=0)
        return min(float(v)/(max_bound-min_bound)+min_bound, 1.0)
    def getValueFromBool(b):
        if b:
            return 1.0
        return 0.0
    try:
        board, rep1, rep2, mcnt, npcnt, cast11, cast12, cast21, cast22, _ = canonicalBoard
    except ValueError:
        board, rep1, rep2, mcnt, npcnt, cast11, cast12, cast21, cast22 = canonicalBoard
    board = chess.Board(board)
    piece_features = getPieces(board)
    # threats_features = getThreats(board)
    rep1_features = getValuePlane(getNorm(rep1))
    rep2_features = getValuePlane(getNorm(rep2))
    mcnt_features = getValuePlane(getNorm(mcnt, max_bound=max_moves))
    npcnt_features = getValuePlane(getNorm(npcnt, max_bound=max_no_progress))
    cast11_features = getValuePlane(getValueFromBool(cast11))
    cast12_features = getValuePlane(getValueFromBool(cast12))
    cast21_features = getValuePlane(getValueFromBool(cast21))
    cast22_features = getValuePlane(getValueFromBool(cast22))
    # all_features = numpy.concatenate((piece_features, threats_features, rep1_features, rep2_features, mcnt_features, npcnt_features, cast11_features, cast12_features, cast21_features, cast22_features), axis=-1)
    all_features = numpy.concatenate((piece_features, rep1_features, rep2_features, mcnt_features, npcnt_features, cast11_features, cast12_features, cast21_features, cast22_features), axis=-1)
    return all_features

awsize = 81

def getActionFeatures(game, board, action):
    s = game.action_list[action]
    st_pos = (ord(s[0])-ord('a'))+(ord(s[1])-ord('1'))*8
    en_pos = (ord(s[2])-ord('a'))+(ord(s[3])-ord('1'))*8
    is_black = (board[-1]==-1)
    # print(board)
    b = game.restoreOriginBoard(board)
    
    # try:
    st_piece = b.piece_at(st_pos).symbol()
    en_piece = b.piece_at(en_pos)
    # except Exception:
    #     print(b.unicode().replace(u'·', u'.'))
    #     print(st_pos, en_pos, s, is_black)
    if en_piece:
        en_piece = en_piece.symbol()
    else:
        en_piece = 'null'
    pro = 'null'
    if len(s)>4:
        pro=s[4]
    b.push(chess.Move.from_uci(game.action_list[action]))
    check = b.is_check()
    # pos 0~63
    # color 64-null 65-white 66-black
    if is_black:
        color_sheet = {'null':64, 'black':65, 'white':66}
    else:
        color_sheet = {'null':64, 'white':65, 'black':66}
    # piece 67-null 68-p 69-r 70-n 71-b 72-q 73-k
    piece_sheet = {'null':67, 'p':68, 'r':69, 'n':70, 'b':71, 'q':72, 'k':73}
    # promote 74-null 75-q 76-b 77-n 78-r
    promote_sheet = {'null':74, 'q':75, 'b':76, 'n':77, 'r':78}
    # check 79-yes 80-no
    check_sheet = {True:79, False:80}
    ret = []
    if st_piece.lower()==st_piece:
        ret.append(color_sheet['black'])
    else:
        ret.append(color_sheet['white'])
    ret.append(piece_sheet[st_piece.lower()])
    if is_black:
        st_pos = st_pos%8+(7-st_pos//8)*8
        en_pos = en_pos%8+(7-en_pos//8)*8  # COPY is cheap, but dangerous!
    # activate this in next training!!!!!
    ret.append(st_pos)
    ret.append(en_pos)
    if en_piece=='null':
        ret.append(color_sheet['null'])
        ret.append(piece_sheet['null'])
    else:
        if en_piece.lower()==en_piece:
            ret.append(color_sheet['black'])
        else:
            ret.append(color_sheet['white'])
        ret.append(piece_sheet[en_piece.lower()])
    ret.append(promote_sheet[pro])
    ret.append(check_sheet[check])
    # print(ret)

    return numpy.array(ret, numpy.int32)

def extend2Vector(l, n):
    if len(l)==n:
        return l
    v = numpy.zeros(n)
    for i in l:
        v[i]=1.0
    return v

