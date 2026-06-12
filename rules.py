from board import in_bounds


def get_valid_moves(board,y,x):
    if not in_bounds(x, y):
        return []
    piece=board[y][x]
    if piece == '.':
        return []
    if piece[-1]=='c':
        return che_moves(board,y,x)
    if piece[-1]=='m':
        return ma_moves(board,y,x)
    if piece[-1]=='p':
        return pao_moves(board,y,x)
    if piece[-1]=='z':
        return bing_moves(board,y,x)
    if piece[-1]=='j':
        return jiang_moves(board,y,x)
    if piece[-1]=='s':
        return shi_moves(board, y, x)
    if piece[-1]=='x':
        return xiang_moves(board, y, x)
    return []

def get_legal_moves(board_obj, x, y):
    board = board_obj.board
    if not in_bounds(x, y):
        return []
    piece = board[y][x]
    if piece == '.':
        return []
    raw_moves = get_valid_moves(board, y, x)
    side = piece[0]
    legal_moves = []
    for nx, ny in raw_moves:
        captured = None
        try:
            captured = board_obj.move(x, y, nx, ny)
            ok = (not is_in_check(board_obj.board, side)and not face_to_face(board_obj.board))
        finally:
            if captured is not None:
                board_obj.undo_move(x, y, nx, ny, captured)
        if ok:
            legal_moves.append((nx, ny))
    return legal_moves

def has_any_legal_moves(board_obj, side):
    board = board_obj.board
    for y in range(10):
        for x in range(9):
            piece = board[y][x]
            if piece == '.':
                continue
            if piece[0] != side:
                continue
            moves = get_legal_moves(board_obj, x, y)
            if moves:
                return True
    return False
def get_game_state(board_obj, side):
    if find_jiang(board_obj.board, side) is None:
        return "checkmate"
    in_check = is_in_check(board_obj.board, side)
    has_moves = has_any_legal_moves(board_obj, side)
    if has_moves:
        return None
    if in_check:
        return "checkmate"
    return "stalemate"

def find_jiang(board, side):
    if side not in ('r', 'b'):
        return None
    target = side + '_j'
    for y in range(10):
        for x in range(9):
            if board[y][x] == target:
                return x, y
    return None

#是否被攻击
def is_attacked(board, x, y, attacker_side):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        blockers = 0
        while 0 <= nx < 9 and 0 <= ny < 10:
            piece = board[ny][nx]
            if piece != '.':
                if piece[0] == attacker_side:
                    ptype = piece[-1]
                    if blockers == 0 and ptype == 'c':
                        return True
                    if blockers == 0 and ptype == 'j' and dx == 0:
                        return True
                    if blockers == 1 and ptype == 'p':
                        return True
                blockers += 1
                if blockers > 1:
                    break
            nx += dx
            ny += dy

    for hx, hy in (
        (x - 2, y - 1), (x - 2, y + 1),
        (x + 2, y - 1), (x + 2, y + 1),
        (x - 1, y - 2), (x + 1, y - 2),
        (x - 1, y + 2), (x + 1, y + 2),
    ):
        if not (0 <= hx < 9 and 0 <= hy < 10):
            continue
        if board[hy][hx] != attacker_side + '_m':
            continue
        dx = x - hx
        dy = y - hy
        if abs(dx) == 2:
            leg_x, leg_y = hx + (1 if dx > 0 else -1), hy
        else:
            leg_x, leg_y = hx, hy + (1 if dy > 0 else -1)
        if board[leg_y][leg_x] == '.':
            return True

    if attacker_side == 'b':
        py = y - 1
        if 0 <= py < 10 and board[py][x] == 'b_z':
            return True
        for px in (x - 1, x + 1):
            if 0 <= px < 9 and y >= 5 and board[y][px] == 'b_z':
                return True
    else:
        py = y + 1
        if 0 <= py < 10 and board[py][x] == 'r_z':
            return True
        for px in (x - 1, x + 1):
            if 0 <= px < 9 and y <= 4 and board[y][px] == 'r_z':
                return True

    for ax, ay in ((x - 1, y - 1), (x + 1, y - 1),
                   (x - 1, y + 1), (x + 1, y + 1)):
        if not (0 <= ax < 9 and 0 <= ay < 10):
            continue
        if board[ay][ax] != attacker_side + '_s':
            continue
        if attacker_side == 'b' and 0 <= y <= 2 and 3 <= x <= 5:
            return True
        if attacker_side == 'r' and 7 <= y <= 9 and 3 <= x <= 5:
            return True

    for ex, ey in ((x - 2, y - 2), (x + 2, y - 2),
                   (x - 2, y + 2), (x + 2, y + 2)):
        if not (0 <= ex < 9 and 0 <= ey < 10):
            continue
        if board[ey][ex] != attacker_side + '_x':
            continue
        eye_x = (ex + x) // 2
        eye_y = (ey + y) // 2
        if board[eye_y][eye_x] != '.':
            continue
        if attacker_side == 'b' and y <= 4:
            return True
        if attacker_side == 'r' and y >= 5:
            return True

    for jx, jy in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
        if not (0 <= jx < 9 and 0 <= jy < 10):
            continue
        if board[jy][jx] != attacker_side + '_j':
            continue
        if attacker_side == 'b' and 0 <= y <= 2 and 3 <= x <= 5:
            return True
        if attacker_side == 'r' and 7 <= y <= 9 and 3 <= x <= 5:
            return True
    return False

#是否将军
def is_in_check(board, side):
    pos = find_jiang(board, side)
    if pos is None:
        return False
    x, y = pos
    enemy = 'b' if side == 'r' else 'r'
    return is_attacked(board, x, y, enemy)

#将军碰面
def face_to_face(board):
    rpos = find_jiang(board, 'r')
    bpos = find_jiang(board, 'b')
    if rpos is None or bpos is None:
        return False
    rx, ry = rpos
    bx, by = bpos
    if rx != bx:
        return False
    top = min(ry, by)
    bottom = max(ry, by)
    for y in range(top + 1, bottom):
        if board[y][rx] != '.':
            return False
    return True

def che_moves(board, y, x):
    moves = []
    piece = board[y][x]
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        while 0 <= ny < 10 and 0 <= nx < 9:
            target = board[ny][nx]
            if target == '.':
                moves.append((nx, ny))
            else:
                # 有子 → 判断敌我
                if piece[0]!= target[0]:
                    moves.append((nx, ny))
                break
            ny += dy
            nx += dx
    return moves


def ma_moves(board, y, x):
    moves = []
    piece = board[y][x]
    patterns = [
        ((1, 0), (2, 1)),
        ((1, 0), (2, -1)),
        ((-1, 0), (-2, 1)),
        ((-1, 0), (-2, -1)),
        ((0, 1), (1, 2)),
        ((0, 1), (-1, 2)),
        ((0, -1), (1, -2)),
        ((0, -1), (-1, -2)),
    ]
    for (ly, lx), (dy, dx) in patterns:
        leg_y = y + ly
        leg_x = x + lx
        if not (0 <= leg_y < 10 and 0 <= leg_x < 9):
            continue
        if board[leg_y][leg_x] != '.':
            continue
        ny = y + dy
        nx = x + dx
        if 0 <= ny < 10 and 0 <= nx < 9:
            target = board[ny][nx]
            if target == '.' or piece[0] != target[0]:
                moves.append((nx, ny))
    return moves


def pao_moves(board, y, x):
    moves = []
    piece = board[y][x]
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        jumped = False   # 是否已经跳过一个子（炮架）
        while 0 <= ny < 10 and 0 <= nx < 9:
            target = board[ny][nx]
            if not jumped:
                # 还没遇到炮架
                if target == '.':
                    moves.append((nx, ny))  # 可以走
                else:
                    jumped = True  # 找到炮架
            else:
                # 已经有炮架 → 只能吃
                if target != '.':
                    if piece[0] != target[0]:
                        moves.append((nx, ny))  # 吃
                    break  # 无论如何都停
            ny += dy
            nx += dx
    return moves


def bing_moves(board, y, x):
    moves = []
    piece = board[y][x]
    side = piece[0]
    directions = []
    if side == 'r':
        directions.append((-1, 0))
        if y <= 4:  # 过河
            directions += [(0, 1), (0, -1)]
    else:
        directions.append((1, 0))
        if y >= 5:
            directions += [(0, 1), (0, -1)]
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < 10 and 0 <= nx < 9:
            target = board[ny][nx]
            if target == '.' or piece[0] != target[0]:
                moves.append((nx, ny))
    return moves


def jiang_moves(board, y, x):
    moves = []
    piece = board[y][x]
    side = piece[0]
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= nx < 9:
            if side == 'r' and 7 <= ny <= 9 and 3 <= nx <= 5:
                target = board[ny][nx]
                if target == '.' or piece[0] != target[0]:
                    moves.append((nx, ny))
            if side == 'b' and 0 <= ny <= 2 and 3 <= nx <= 5:
                target = board[ny][nx]
                if target == '.' or piece[0] != target[0]:
                    moves.append((nx, ny))

    return moves

def shi_moves(board, y, x):
    moves = []
    piece = board[y][x]
    side = piece[0]
    directions = [(1,1), (1,-1), (-1,1), (-1,-1)]
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= nx < 9:
            if side == 'r' and 7 <= ny <= 9 and 3 <= nx <= 5:
                target = board[ny][nx]
                if target == '.' or piece[0] != target[0]:
                    moves.append((nx, ny))

            if side == 'b' and 0 <= ny <= 2 and 3 <= nx <= 5:
                target = board[ny][nx]
                if target == '.' or piece[0] != target[0]:
                    moves.append((nx, ny))
    return moves

def xiang_moves(board, y, x):
    moves = []
    piece = board[y][x]
    side = piece[0]
    patterns = [
        ((1,1), (2,2)),
        ((1,-1), (2,-2)),
        ((-1,1), (-2,2)),
        ((-1,-1), (-2,-2)),
    ]
    for (ey, ex), (dy, dx) in patterns:
        eye_y = y + ey
        eye_x = x + ex
        if not (0 <= eye_y < 10 and 0 <= eye_x < 9):
            continue
        # 象眼被堵
        if board[eye_y][eye_x] != '.':
            continue
        ny = y + dy
        nx = x + dx
        if 0 <= ny < 10 and 0 <= nx < 9:
            # 不能过河
            if side == 'r' and ny < 5:
                continue
            if side == 'b' and ny > 4:
                continue
            target = board[ny][nx]
            if target == '.' or piece[0] != target[0]:
                moves.append((nx, ny))
    return moves
