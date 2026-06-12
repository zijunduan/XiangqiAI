import random
import math
import copy
import time
import os
from rules import get_legal_moves, is_in_check, get_valid_moves, find_jiang, is_attacked

# ─────────────────────────────────────────────────────────────────────────────
#  棋子基础价值
# ─────────────────────────────────────────────────────────────────────────────
piece_values = {
    'j': 10000,
    'c': 900,
    'p': 450,
    'm': 400,
    'x': 200,
    's': 200,
    'z': 100,
}

MATE_SCORE = 999999
PST_EVAL_WEIGHT = 0.45
PST_ORDER_WEIGHT = 0.45


def opponent(side):
    return 'r' if side == 'b' else 'b'


def terminal_board_score(board, ply=0):
    if find_jiang(board, 'b') is None:
        return -MATE_SCORE + ply
    if find_jiang(board, 'r') is None:
        return MATE_SCORE - ply
    return None


def captured_king_score(captured, ply=0):
    if captured == 'r_j':
        return MATE_SCORE - ply
    if captured == 'b_j':
        return -MATE_SCORE + ply
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  位置价值表（PST）—— 黑方视角；红方使用时行索引取 9-y（镜像）
# ─────────────────────────────────────────────────────────────────────────────
pawn_table_b = [          # 黑兵：越往红方腹地越高
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],  # 过河线（y=4/5分界）
    [20, 20, 30, 50, 60, 50, 30, 20, 20],  # 过河后横向可走，价值大增
    [30, 30, 40, 65, 75, 65, 40, 30, 30],
    [45, 45, 55, 80, 90, 80, 55, 45, 45],
    [55, 55, 65, 90,100, 90, 65, 55, 55],
    [60, 60, 70, 95,105, 95, 70, 60, 60],
]
knight_table = [
    [-10,-10,-10,-10,-10,-10,-10,-10,-10],
    [-10,  0,  5, 10, 10, 10,  5,  0,-10],
    [-10,  5, 10, 15, 15, 15, 10,  5,-10],
    [-10, 10, 15, 20, 20, 20, 15, 10,-10],
    [-10, 10, 15, 20, 25, 20, 15, 10,-10],
    [-10, 10, 15, 20, 20, 20, 15, 10,-10],
    [-10,  5, 10, 15, 15, 15, 10,  5,-10],
    [-10,  0,  5, 10, 10, 10,  5,  0,-10],
    [-10,-10,-10,-10,-10,-10,-10,-10,-10],
    [-10,-20,-10,-10,-10,-10,-10,-10,-10],
]
rook_table = [
    [20, 20, 30, 40, 40, 40, 30, 20, 20],
    [20, 30, 40, 50, 50, 50, 40, 30, 20],
    [30, 40, 50, 60, 60, 60, 50, 40, 30],
    [40, 50, 60, 70, 70, 70, 60, 50, 40],
    [50, 60, 70, 80, 80, 80, 70, 60, 50],
    [50, 60, 70, 80, 80, 80, 70, 60, 50],
    [40, 50, 60, 70, 70, 70, 60, 50, 40],
    [30, 40, 50, 60, 60, 60, 50, 40, 30],
    [20, 30, 40, 50, 50, 50, 40, 30, 20],
    [20, 20, 30, 40, 40, 40, 30, 20, 20],
]
cannon_table = [
    [ 0,  0, 10, 20, 20, 20, 10,  0,  0],
    [ 0, 10, 20, 30, 30, 30, 20, 10,  0],
    [10, 20, 30, 40, 40, 40, 30, 20, 10],
    [20, 30, 40, 50, 50, 50, 40, 30, 20],
    [30, 40, 50, 60, 60, 60, 50, 40, 30],
    [30, 40, 50, 60, 60, 60, 50, 40, 30],
    [20, 30, 40, 50, 50, 50, 40, 30, 20],
    [10, 20, 30, 40, 40, 40, 30, 20, 10],
    [ 0, 10, 20, 30, 30, 30, 20, 10,  0],
    [ 0,  0, 10, 20, 20, 20, 10,  0,  0],
]
king_table = [
    [-50,-40,-30,-20,-20,-20,-30,-40,-50],
    [-40,-30,-20,-10,-10,-10,-20,-30,-40],
    [-30,-20,-10,  0,  0,  0,-10,-20,-30],
    [-20,-10,  0, 10, 10, 10,  0,-10,-20],
    [-20,-10,  0, 10, 20, 10,  0,-10,-20],
    [-20,-10,  0, 10, 10, 10,  0,-10,-20],
    [-30,-20,-10,  0,  0,  0,-10,-20,-30],
    [-40,-30,-20,-10,-10,-10,-20,-30,-40],
    [-50,-40,-30,-20,-20,-20,-30,-40,-50],
    [-60,-50,-40,-30,-30,-30,-40,-50,-60],
]
elephant_table = [
    [ 0,  0,  5,  0, 10,  0,  5,  0,  0],
    [ 0,  5, 10, 10, 15, 10, 10,  5,  0],
    [ 5, 10, 15, 20, 20, 20, 15, 10,  5],
    [10, 10, 20, 25, 30, 25, 20, 10, 10],
    [10, 15, 20, 30, 35, 30, 20, 15, 10],
    [10, 10, 20, 25, 30, 25, 20, 10, 10],
    [ 5, 10, 15, 20, 20, 20, 15, 10,  5],
    [ 0,  5, 10, 10, 15, 10, 10,  5,  0],
    [ 0,  0,  5,  0, 10,  0,  5,  0,  0],
    [ 0,  0,  0,  0,  5,  0,  0,  0,  0],
]
advisor_table = [
    [ 0,  0, 10,  0, 10,  0, 10,  0,  0],
    [ 0, 10, 20, 20, 20, 20, 20, 10,  0],
    [10, 20, 30, 30, 30, 30, 30, 20, 10],
    [10, 20, 30, 40, 40, 40, 30, 20, 10],
    [10, 20, 30, 40, 50, 40, 30, 20, 10],
    [10, 20, 30, 40, 40, 40, 30, 20, 10],
    [10, 20, 30, 30, 30, 30, 30, 20, 10],
    [ 0, 10, 20, 20, 20, 20, 20, 10,  0],
    [ 0,  0, 10,  0, 10,  0, 10,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
]

pst = {
    'j': king_table,
    's': advisor_table,
    'x': elephant_table,
    'm': knight_table,
    'c': rook_table,
    'p': cannon_table,
    'z': pawn_table_b,
}

# ─────────────────────────────────────────────────────────────────────────────
#  Zobrist Hash —— 比 tuple hash 快约 5-8 倍
# ─────────────────────────────────────────────────────────────────────────────
_PIECES = ['b_c','b_m','b_x','b_s','b_j','b_p','b_z',
           'r_c','r_m','r_x','r_s','r_j','r_p','r_z']
_PIECE_IDX = {p: i for i, p in enumerate(_PIECES)}

random.seed(20250604)
ZOBRIST_TABLE = [
    [random.getrandbits(64) for _ in range(14)]
    for _ in range(90)
]
random.seed()  # 恢复随机


def compute_hash(board):
    h = 0
    for y in range(10):
        for x in range(9):
            p = board[y][x]
            if p != '.':
                h ^= ZOBRIST_TABLE[y * 9 + x][_PIECE_IDX[p]]
    return h


# ─────────────────────────────────────────────────────────────────────────────
#  工具
# ─────────────────────────────────────────────────────────────────────────────

def get_all_moves(board_obj, side):
    board = board_obj.board
    moves = []
    for y in range(10):
        for x in range(9):
            p = board[y][x]
            if p == '.' or p[0] != side:
                continue
            for nx, ny in get_legal_moves(board_obj, x, y):
                moves.append((x, y, nx, ny))
    return moves


def get_capture_moves(board_obj, side):
    """只返回吃子着法，用于静态搜索"""
    board = board_obj.board
    moves = []
    for y in range(10):
        for x in range(9):
            p = board[y][x]
            if p == '.' or p[0] != side:
                continue
            for nx, ny in get_legal_moves(board_obj, x, y):
                if board[ny][nx] != '.':
                    moves.append((x, y, nx, ny))
    return moves


def _pseudo_attacks(board, side):
    attacks = set()
    for y in range(10):
        for x in range(9):
            p = board[y][x]
            if p == '.' or p[0] != side:
                continue
            attacks.update(get_valid_moves(board, y, x))
    return attacks


def _pseudo_mobility(board, side):
    total = 0
    for y in range(10):
        for x in range(9):
            p = board[y][x]
            if p == '.' or p[0] != side:
                continue
            total += len(get_valid_moves(board, y, x))
    return total


def _attack_map_and_mobility(board, side):
    attacks = set()
    mobility = 0
    for y in range(10):
        for x in range(9):
            p = board[y][x]
            if p == '.' or p[0] != side:
                continue
            moves = get_valid_moves(board, y, x)
            mobility += len(moves)
            attacks.update(moves)
    return attacks, mobility


def _is_defended(board, x, y, side):
    target = board[y][x]
    board[y][x] = '.'
    try:
        for iy in range(10):
            for ix in range(9):
                p = board[iy][ix]
                if p == '.' or p[0] != side:
                    continue
                if (x, y) in get_valid_moves(board, iy, ix):
                    return True
        return False
    finally:
        board[y][x] = target


def _hanging_piece_score(board, black_attacks=None, red_attacks=None):
    if black_attacks is None:
        black_attacks = _pseudo_attacks(board, 'b')
    if red_attacks is None:
        red_attacks = _pseudo_attacks(board, 'r')
    score = 0
    for y in range(10):
        for x in range(9):
            p = board[y][x]
            if p == '.' or p[-1] == 'j':
                continue
            value = piece_values[p[-1]]
            pos = (x, y)
            if p[0] == 'b':
                if pos in red_attacks:
                    score -= value * (0.14 if pos not in black_attacks else 0.05)
            else:
                if pos in black_attacks:
                    score += value * (0.14 if pos not in red_attacks else 0.05)
    return int(score)


def _direct_hanging_score(board):
    score = 0
    for y in range(10):
        for x in range(9):
            piece = board[y][x]
            if piece == '.' or piece[-1] == 'j':
                continue
            side = piece[0]
            enemy = opponent(side)
            value = piece_values[piece[-1]]
            if not is_attacked(board, x, y, enemy):
                continue
            defended = is_attacked(board, x, y, side)
            penalty = value * (0.22 if not defended else 0.08)
            if value >= 400 and not defended:
                penalty += 70
            if side == 'b':
                score -= penalty
            else:
                score += penalty
    return int(score)


def evaluate_minimax_static(board):
    return evaluate_fast(board) + _direct_hanging_score(board)


def _king_file_pressure(board):
    score = 0
    for side in ('b', 'r'):
        king = side + '_j'
        kx = ky = None
        for y in range(10):
            for x in range(9):
                if board[y][x] == king:
                    kx, ky = x, y
                    break
            if kx is not None:
                break
        if kx is None:
            continue
        enemy = opponent(side)
        pressure = 0
        for y in range(10):
            p = board[y][kx]
            if p != '.' and p[0] == enemy and p[-1] in ('c', 'p'):
                pressure += 1
        score += (-35 if side == 'b' else 35) * pressure
    return score


def _non_king_material(board):
    total = 0
    for row in board:
        for p in row:
            if p != '.' and p[-1] != 'j':
                total += piece_values[p[-1]]
    return total


def _side_non_king_material(board, side):
    total = 0
    for row in board:
        for p in row:
            if p != '.' and p[0] == side and p[-1] != 'j':
                total += piece_values[p[-1]]
    return total


def _king_line_threat_score(board):
    score = 0
    for side in ('b', 'r'):
        king = side + '_j'
        kx = ky = None
        for y in range(10):
            for x in range(9):
                if board[y][x] == king:
                    kx, ky = x, y
                    break
            if kx is not None:
                break
        if kx is None:
            continue

        enemy = opponent(side)
        pressure = 0
        for dy in (-1, 1):
            blockers = 0
            y = ky + dy
            while 0 <= y < 10:
                p = board[y][kx]
                if p != '.':
                    if p[0] == enemy:
                        if p[-1] == 'c' and blockers == 0:
                            pressure += 120
                        elif p[-1] == 'p' and blockers == 1:
                            pressure += 100
                        elif p[-1] == 'j' and blockers == 0:
                            pressure += 500
                    blockers += 1
                    if blockers > 1:
                        break
                y += dy
        score += -pressure if side == 'b' else pressure
    return score


def _palace_attack_score(board, black_attacks=None, red_attacks=None):
    if black_attacks is None:
        black_attacks = _pseudo_attacks(board, 'b')
    if red_attacks is None:
        red_attacks = _pseudo_attacks(board, 'r')
    black_palace = [(x, y) for y in range(0, 3) for x in range(3, 6)]
    red_palace = [(x, y) for y in range(7, 10) for x in range(3, 6)]
    score = 0
    score += sum(18 for sq in red_palace if sq in black_attacks)
    score -= sum(18 for sq in black_palace if sq in red_attacks)
    return score


# ─────────────────────────────────────────────────────────────────────────────
#  评估函数
# ─────────────────────────────────────────────────────────────────────────────

def _pst_val(ptype, y, x, side):
    """返回某棋子在 (y,x) 位置的 PST 分，已按阵营镜像"""
    row = (9 - y) if side == 'r' else y
    return pst[ptype][row][x]


PAWN_FILE_SHAPE = [-38, -16, 4, 18, 28, 18, 4, -16, -38]


def _pawn_crossed(side, y):
    return y >= 5 if side == 'b' else y <= 4


def _pawn_forward(side, sy, ey):
    return ey > sy if side == 'b' else ey < sy


def _pawn_progress(side, y):
    return y if side == 'b' else 9 - y


def _pawn_in_enemy_palace(side, x, y):
    if not 3 <= x <= 5:
        return False
    return y >= 7 if side == 'b' else y <= 2


def _pawn_final_rank(side, y):
    return y == 9 if side == 'b' else y == 0


def _pawn_move_shape_delta(side, sx, sy, ex, ey,
                           is_capture=False, gives_check=False):
    if not (_pawn_crossed(side, sy) or _pawn_crossed(side, ey)):
        return 0

    before_dist = abs(sx - 4)
    after_dist = abs(ex - 4)
    toward_center = before_dist - after_dist
    forward = _pawn_forward(side, sy, ey)
    sideways = (ey == sy)
    progress_after = _pawn_progress(side, ey)
    score = 0

    if forward:
        if progress_after <= 6:
            score += 18
        elif progress_after == 7:
            score += 10
        elif progress_after == 8:
            score += 4 if _pawn_in_enemy_palace(side, ex, ey) else -8
        else:
            score -= 18
    elif sideways:
        score -= 8

    if toward_center > 0:
        score += toward_center * 20
        if sideways:
            score += 12
    elif toward_center < 0:
        score += toward_center * 30
        if sideways:
            score -= 24

    score += PAWN_FILE_SHAPE[ex]
    if ex in (0, 8):
        score -= 26 + _pawn_progress(side, ey) * 4
    elif ex in (1, 7):
        score -= 8

    enemy_palace_y = range(7, 10) if side == 'b' else range(0, 3)
    if ey in enemy_palace_y and 3 <= ex <= 5:
        score += 34
    if progress_after >= 8:
        if _pawn_final_rank(side, ey):
            score -= 18 if _pawn_in_enemy_palace(side, ex, ey) else 34
        elif not _pawn_in_enemy_palace(side, ex, ey):
            score -= 12

    if is_capture:
        score = int(score * 0.45) + 18
    if gives_check:
        score = max(score, 10) + 20
    return score


def _pawn_move_shape_score(board, move, side=None,
                           gives_check=False):
    sx, sy, ex, ey = move
    piece = board[sy][sx]
    if piece == '.' or piece[-1] != 'z':
        return 0
    side = side or piece[0]
    target = board[ey][ex]
    return _pawn_move_shape_delta(
        side, sx, sy, ex, ey,
        is_capture=(target != '.'),
        gives_check=gives_check,
    )


def _side_sign(side):
    return 1 if side == 'b' else -1


def _blockers_between(board, x, y1, y2):
    top = min(y1, y2)
    bottom = max(y1, y2)
    blockers = 0
    for y in range(top + 1, bottom):
        if board[y][x] != '.':
            blockers += 1
    return blockers


def _leg_block_count(board, x, y):
    count = 0
    for lx, ly in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if 0 <= lx < 9 and 0 <= ly < 10 and board[ly][lx] != '.':
            count += 1
    return count


def _cheap_structure_score(board, pieces, counts, kings):
    score = 0
    for side in ('b', 'r'):
        sign = _side_sign(side)
        enemy = opponent(side)
        enemy_king = kings.get(enemy)

        advisors = counts[side].get('s', 0)
        elephants = counts[side].get('x', 0)
        score += sign * (advisors - 2) * 22
        score += sign * (elephants - 2) * 18
        if advisors == 0:
            score -= sign * 36
        if elephants == 0:
            score -= sign * 28

        king_pos = kings.get(side)
        if king_pos:
            kx, ky = king_pos
            home_y = 0 if side == 'b' else 9
            forward_steps = abs(ky - home_y)
            score -= sign * forward_steps * 58
            if kx != 4:
                score -= sign * 18
            if forward_steps == 0 and kx == 4:
                score += sign * 18

        pawn_positions = [
            (px, py) for px, py, pt in pieces[side]
            if pt == 'z'
        ]

        for x, y, ptype in pieces[side]:
            if ptype == 'c':
                own_on_file = 0
                enemy_on_file = 0
                for fy in range(10):
                    piece = board[fy][x]
                    if piece == '.':
                        continue
                    if piece[0] == side:
                        own_on_file += 1
                    elif piece[-1] != 'j':
                        enemy_on_file += 1
                if own_on_file <= 1:
                    score += sign * 26
                    if enemy_on_file == 0:
                        score += sign * 12
                if enemy_king and x == enemy_king[0]:
                    blockers = _blockers_between(board, x, y, enemy_king[1])
                    if blockers == 0:
                        score += sign * 130
                    elif blockers == 1:
                        score += sign * 42

            elif ptype == 'p':
                if enemy_king and x == enemy_king[0]:
                    blockers = _blockers_between(board, x, y, enemy_king[1])
                    if blockers == 1:
                        score += sign * 105
                    elif blockers == 2:
                        score += sign * 28

            elif ptype == 'm':
                score -= sign * _leg_block_count(board, x, y) * 16
                if 2 <= x <= 6 and 2 <= y <= 7:
                    score += sign * 12

            elif ptype == 'z':
                crossed = (side == 'b' and y >= 5) or (side == 'r' and y <= 4)
                if crossed:
                    progress = _pawn_progress(side, y)
                    in_enemy_palace = _pawn_in_enemy_palace(side, x, y)
                    score += sign * 18
                    score += sign * PAWN_FILE_SHAPE[x]
                    if 3 <= x <= 5:
                        score += sign * 10
                    if x in (0, 8):
                        score -= sign * (28 + _pawn_progress(side, y) * 5)
                    elif x in (1, 7):
                        score -= sign * 10
                    near_friend = any(
                        (px, py) != (x, y) and
                        abs(px - x) <= 2 and abs(py - y) <= 1
                        for px, py in pawn_positions
                    )
                    if not near_friend and x in (0, 1, 7, 8):
                        score -= sign * 14
                    if progress >= 8:
                        if _pawn_final_rank(side, y):
                            score -= sign * (24 if in_enemy_palace else 46)
                        elif not in_enemy_palace:
                            score -= sign * 18
                        if not near_friend:
                            score -= sign * 10
                enemy_palace_y = range(7, 10) if side == 'b' else range(0, 3)
                if y in enemy_palace_y and 3 <= x <= 5:
                    score += sign * 26
                for dx in (-1, 1):
                    nx = x + dx
                    if 0 <= nx < 9 and board[y][nx] == side + '_z':
                        score += sign * 6

    return score


def evaluate_fast(board):
    """轻量评估：用于 minimax 的大量叶节点，避免反复生成攻击图。"""
    score = 0
    pieces = {'b': [], 'r': []}
    counts = {'b': {}, 'r': {}}
    kings = {}
    for y in range(10):
        for x in range(9):
            p = board[y][x]
            if p == '.':
                continue
            side = p[0]
            ptype = p[-1]
            pieces[side].append((x, y, ptype))
            counts[side][ptype] = counts[side].get(ptype, 0) + 1
            if ptype == 'j':
                kings[side] = (x, y)
            v = piece_values[ptype] + int(
                _pst_val(ptype, y, x, side) * PST_EVAL_WEIGHT)
            if side == 'b':
                score += v
            else:
                score -= v
    if 'b' not in kings:
        return -MATE_SCORE
    if 'r' not in kings:
        return MATE_SCORE
    score += _cheap_structure_score(board, pieces, counts, kings)
    score += _king_file_pressure(board)
    score += _king_line_threat_score(board)
    return score


def evaluate(board):
    """
    完整静态评估，正值黑方优，负值红方优。
    包含：棋子价值 + PST + 攻击图/机动性/悬子/王线压力。
    """
    score = evaluate_fast(board)
    black_attacks, black_mobility = _attack_map_and_mobility(board, 'b')
    red_attacks, red_mobility = _attack_map_and_mobility(board, 'r')
    score += (black_mobility - red_mobility) * 2
    score += _hanging_piece_score(board, black_attacks, red_attacks)
    score += _palace_attack_score(board, black_attacks, red_attacks)
    return score


def evaluate_full(board, board_obj):
    """带机动性 + 将军奖励的完整评估（根节点 / 叶节点使用）"""
    score = evaluate(board)

    # 将军奖励
    if is_in_check(board, 'r'):
        score += 60
    if is_in_check(board, 'b'):
        score -= 60

    # 机动性（合法着法数之差）× 权重 3
    b_mob = len(get_all_moves(board_obj, 'b'))
    r_mob = len(get_all_moves(board_obj, 'r'))
    score += (b_mob - r_mob) * 3

    return score


# ─────────────────────────────────────────────────────────────────────────────
#  着法排序分（MVV-LVA + 杀手 + PST 增益）
# ─────────────────────────────────────────────────────────────────────────────

def move_priority(board, move, killers):
    sx, sy, ex, ey = move
    target = board[ey][ex]
    attacker = board[sy][sx]
    score = 0
    if target != '.':
        # MVV-LVA：大子被吃排最前；用小子吃大子额外加分
        score += piece_values[target[-1]] * 10 - piece_values[attacker[-1]]
    elif move in killers:
        score += 80                      # 杀手着法提权
    # PST 增益
    score += int((_pst_val(attacker[-1], ey, ex, attacker[0]) -
                  _pst_val(attacker[-1], sy, sx, attacker[0])) *
                 PST_ORDER_WEIGHT)
    score += _pawn_move_shape_score(board, move) * 2
    return score


def move_gives_check(board_obj, move, side):
    sx, sy, ex, ey = move
    captured = None
    try:
        captured = board_obj.move(sx, sy, ex, ey)
        return is_in_check(board_obj.board, opponent(side))
    finally:
        if captured is not None:
            board_obj.undo_move(sx, sy, ex, ey, captured)


# ─────────────────────────────────────────────────────────────────────────────
#  置换表常量
# ─────────────────────────────────────────────────────────────────────────────
TT_EXACT = 0
TT_LOWER = 1   # alpha 下界
TT_UPPER = 2   # beta 上界


# ─────────────────────────────────────────────────────────────────────────────
#  MinimaxAI（Alpha-Beta + 置换表 + 杀手着法 + 迭代加深 + 静态搜索）
# ─────────────────────────────────────────────────────────────────────────────

def _select_neural_model_path():
    for path in ("chess_model_best.pt", "chess_model.pt"):
        if os.path.exists(path):
            return path
    return "chess_model_best.pt"


NEURAL_MODEL_PATH = _select_neural_model_path()
NEURAL_DEVICE = os.environ.get("CHESS_NEURAL_DEVICE", "auto")
NEURAL_SIMULATIONS = 300
NEURAL_POLICY_WEIGHT = 9000
NEURAL_ROOT_VALUE_WEIGHT = 70
NEURAL_LEAF_VALUE_WEIGHT = 60
_neural_advisor = None
_neural_mcts = None


class NeuralAdvisor:
    def __init__(self, model_path=None, device=None, max_cache=4096):
        self.model_path = model_path or NEURAL_MODEL_PATH
        self.device_name = device or NEURAL_DEVICE
        self.max_cache = max_cache
        self.policy_cache = {}
        self.value_cache = {}
        self.ready = False
        self.model = None
        self.torch = None
        self.F = None
        self.board_to_tensor = None
        self.encode_move = None
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            import torch
            import torch.nn.functional as F
            from network import load_model, board_to_tensor, encode_move

            if self.device_name == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device_name
            self.model = load_model(self.model_path, device=device)
            self.model.eval()
            self.torch = torch
            self.F = F
            self.board_to_tensor = board_to_tensor
            self.encode_move = encode_move
            self.device_name = device
            self.ready = True
            print(f"[neural advisor] loaded {self.model_path} on {device}")
        except Exception as exc:
            print(f"[neural advisor] disabled: {exc}")
            self.ready = False

    def _trim_cache(self, cache):
        if len(cache) > self.max_cache:
            for key in list(cache.keys())[:self.max_cache // 4]:
                cache.pop(key, None)

    def _value_to_black_score(self, value_from_side, side):
        return value_from_side if side == 'b' else -value_from_side

    def value(self, board, side):
        if not self.ready:
            return None
        key = (compute_hash(board), side)
        cached = self.value_cache.get(key)
        if cached is not None:
            return cached
        try:
            tensor = self.board_to_tensor(board, side).unsqueeze(0).to(self.device_name)
            with self.torch.no_grad():
                _, value = self.model(tensor)
            value_black = self._value_to_black_score(float(value.item()), side)
            self.value_cache[key] = value_black
            self._trim_cache(self.value_cache)
            return value_black
        except Exception as exc:
            print(f"[neural advisor] value failed: {exc}")
            self.ready = False
            return None

    def policy_value(self, board, legal_moves, side):
        if not self.ready or not legal_moves:
            return {}, self.value(board, side)
        key = (compute_hash(board), side)
        cached = self.policy_cache.get(key)
        if cached is not None:
            return cached
        try:
            tensor = self.board_to_tensor(board, side).unsqueeze(0).to(self.device_name)
            with self.torch.no_grad():
                logits, value = self.model(tensor)
            idx = self.torch.tensor(
                [self.encode_move(*move) for move in legal_moves],
                dtype=self.torch.long,
                device=self.device_name,
            )
            probs = self.F.softmax(logits[0].index_select(0, idx), dim=0)
            probs = probs.detach().cpu().tolist()
            priors = {move: float(prob) for move, prob in zip(legal_moves, probs)}
            value_black = self._value_to_black_score(float(value.item()), side)
            result = (priors, value_black)
            self.policy_cache[key] = result
            self.value_cache[key] = value_black
            self._trim_cache(self.policy_cache)
            self._trim_cache(self.value_cache)
            return result
        except Exception as exc:
            print(f"[neural advisor] policy failed: {exc}")
            self.ready = False
            return {}, None


def get_neural_advisor():
    global _neural_advisor
    if _neural_advisor is None:
        _neural_advisor = NeuralAdvisor()
    return _neural_advisor if _neural_advisor.ready else None


class MinimaxAI:

    def __init__(self, max_depth=4, time_limit=2.0, use_neural=False):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.use_neural = use_neural
        self.tt = {}                                          # 置换表
        self.killers = [[None, None] for _ in range(32)]     # 杀手着法
        self.history = {}                                     # 历史启发表 {move: score}
        self.eval_cache = {}
        self.check_cache = {}
        self.move_cache = {}
        self.see_cache = {}
        self.nodes = 0
        self.completed_depth = 0
        self.partial_depth = 0
        self.last_root_searched = 0
        self._timeout = False
        self.deadline = None
        self.neural = None
        self.neural_leaf_evals = 0
        self.neural_leaf_budget = 0
        self.repetition_counts = {}
        self.repetition_seen_limit = 2

    # ── 清理 ─────────────────────────────────────────────────────────────────

    def _reset(self):
        self.tt.clear()
        self.killers = [[None, None] for _ in range(32)]
        if len(self.history) > 20000:
            self.history = {
                move: score // 2
                for move, score in self.history.items()
                if score > 1
            }
        self.eval_cache.clear()
        self.check_cache.clear()
        self.move_cache.clear()
        self.see_cache.clear()
        self.nodes = 0
        self.completed_depth = 0
        self._timeout = False
        self.deadline = None
        self.neural = None
        self.neural_leaf_evals = 0
        self.neural_leaf_budget = 0

    def _time_up(self):
        return self.deadline is not None and time.time() >= self.deadline

    def set_repetition_context(self, position_counts=None, seen_limit=2):
        self.repetition_counts = dict(position_counts or {})
        self.repetition_seen_limit = max(1, seen_limit)

    def _position_key(self, board, turn):
        return (turn, tuple(tuple(row) for row in board))

    def _repeat_count_current(self, board, turn):
        if not self.repetition_counts:
            return 0
        return self.repetition_counts.get(self._position_key(board, turn), 0)

    def _repetition_penalty_from_count(self, count):
        if count >= self.repetition_seen_limit:
            return MATE_SCORE // 2
        if count == self.repetition_seen_limit - 1:
            return 420
        return 0

    def _repeat_count_after_move(self, board_obj, move, side):
        sx, sy, ex, ey = move
        captured = board_obj.move(sx, sy, ex, ey)
        try:
            return self._repeat_count_current(board_obj.board, opponent(side))
        finally:
            board_obj.undo_move(sx, sy, ex, ey, captured)

    def _root_repetition_penalty(self, board_obj, move, side):
        count = self._repeat_count_after_move(board_obj, move, side)
        return self._repetition_penalty_from_count(count)

    def _filter_repeating_root_moves(self, board_obj, moves, side):
        if not self.repetition_counts:
            return list(moves)
        non_repeating = [
            move for move in moves
            if self._root_repetition_penalty(board_obj, move, side) < MATE_SCORE // 4
        ]
        return non_repeating or list(moves)

    def _ensure_neural(self):
        if (not self.use_neural or self.time_limit < 0.8 or
                self._timeout or self._time_up()):
            return None
        if self.neural is None:
            self.neural = get_neural_advisor()
            if self.neural is None:
                self.neural_leaf_budget = 0
            elif self.time_limit >= 4.0:
                self.neural_leaf_budget = 96
            elif self.time_limit >= 2.0:
                self.neural_leaf_budget = 48
            else:
                self.neural_leaf_budget = 16
        return self.neural

    def _eval(self, board, side=None, allow_neural=False):
        advisor = self._ensure_neural() if allow_neural and side else None
        neural_key = side if advisor is not None else None
        key = (compute_hash(board), neural_key)
        cached = self.eval_cache.get(key)
        if cached is not None:
            return cached
        value = evaluate_fast(board)
        if (neural_key is not None and
                self.neural_leaf_evals < self.neural_leaf_budget):
            neural_value = advisor.value(board, side)
            if neural_value is not None:
                self.neural_leaf_evals += 1
                weight = NEURAL_LEAF_VALUE_WEIGHT
                if abs(value) >= 900:
                    weight *= 0.45
                value += int(neural_value * weight)
        if len(self.eval_cache) < 200000:
            self.eval_cache[key] = value
        return value

    def _move_gives_check(self, board_obj, move, side):
        key = (compute_hash(board_obj.board), move, side)
        cached = self.check_cache.get(key)
        if cached is not None:
            return cached
        value = move_gives_check(board_obj, move, side)
        if len(self.check_cache) < 200000:
            self.check_cache[key] = value
        return value

    def _get_all_moves(self, board_obj, side):
        key = (compute_hash(board_obj.board), side)
        cached = self.move_cache.get(key)
        if cached is not None:
            return list(cached)
        moves = get_all_moves(board_obj, side)
        if len(self.move_cache) < 100000:
            self.move_cache[key] = tuple(moves)
        return list(moves)

    def _get_capture_moves(self, board_obj, side):
        board = board_obj.board
        return [
            move for move in self._get_all_moves(board_obj, side)
            if board[move[3]][move[2]] != '.'
        ]

    def _least_valuable_attacker(self, board, tx, ty, side):
        best = None
        best_value = float('inf')
        for y in range(10):
            for x in range(9):
                if x == tx and y == ty:
                    continue
                piece = board[y][x]
                if piece == '.' or piece[0] != side:
                    continue
                if (tx, ty) not in get_valid_moves(board, y, x):
                    continue
                value = piece_values[piece[-1]]
                if value < best_value:
                    best_value = value
                    best = (x, y, piece)
        return best

    def _see_capture_sequence(self, board, tx, ty, side, depth=8):
        if depth <= 0:
            return 0
        target = board[ty][tx]
        if target == '.' or target[0] == side:
            return 0

        attacker = self._least_valuable_attacker(board, tx, ty, side)
        if attacker is None:
            return 0

        ax, ay, moving_piece = attacker
        captured_piece = target
        board[ay][ax] = '.'
        board[ty][tx] = moving_piece
        try:
            reply = self._see_capture_sequence(
                board, tx, ty, opponent(side), depth - 1)
            gain = piece_values[captured_piece[-1]] - reply
            return max(0, gain)
        finally:
            board[ay][ax] = moving_piece
            board[ty][tx] = captured_piece

    def _see(self, board_obj, move):
        board = board_obj.board
        sx, sy, ex, ey = move
        moving_piece = board[sy][sx]
        captured_piece = board[ey][ex]
        if moving_piece == '.' or captured_piece == '.':
            return 0

        key = (compute_hash(board), move)
        cached = self.see_cache.get(key)
        if cached is not None:
            return cached

        board_copy = [row[:] for row in board]
        board_copy[sy][sx] = '.'
        board_copy[ey][ex] = moving_piece
        gain = (
            piece_values[captured_piece[-1]] -
            self._see_capture_sequence(
                board_copy, ex, ey, opponent(moving_piece[0]), depth=8)
        )
        if len(self.see_cache) < 100000:
            self.see_cache[key] = gain
        return gain

    def _see_gain_on_square(self, board, tx, ty, side, depth=8):
        target = board[ty][tx]
        if target == '.' or target[0] == side:
            return 0
        return self._see_capture_sequence(board, tx, ty, side, depth)

    def _capture_order_score(self, board_obj, move, force_see=False):
        board = board_obj.board
        sx, sy, ex, ey = move
        attacker = board[sy][sx]
        target = board[ey][ex]
        if attacker == '.' or target == '.':
            return 0

        attacker_value = piece_values[attacker[-1]]
        target_value = piece_values[target[-1]]
        score = target_value * 10 - attacker_value
        if force_see or attacker_value >= target_value:
            see = self._see(board_obj, move)
            score += max(-22000, min(22000, see * 24))
            if see < 0:
                score -= min(16000, -see * 18)
        return score

    def _is_bad_quiescence_capture(self, board_obj, move):
        board = board_obj.board
        sx, sy, ex, ey = move
        attacker = board[sy][sx]
        target = board[ey][ex]
        if attacker == '.' or target == '.':
            return False
        attacker_value = piece_values[attacker[-1]]
        target_value = piece_values[target[-1]]
        if attacker_value <= target_value:
            return False
        return self._see(board_obj, move) < -180

    def _get_noisy_moves(self, board_obj, side, checks_left):
        board = board_obj.board
        captures = [
            move for move in self._get_capture_moves(board_obj, side)
            if not self._is_bad_quiescence_capture(board_obj, move)
        ]
        captures.sort(key=lambda m: self._capture_order_score(board_obj, m),
                      reverse=True)
        if checks_left <= 0:
            return captures

        quiet_checks = []
        for move in self._get_all_moves(board_obj, side):
            sx, sy, ex, ey = move
            if board[ey][ex] != '.':
                continue
            if self._move_gives_check(board_obj, move, side):
                quiet_checks.append(move)
        quiet_checks.sort(key=lambda m: move_priority(board, m, (None, None)),
                          reverse=True)
        return captures + quiet_checks[:4]

    def _root_tactical_bias(self, board_obj, move, side, captured):
        board = board_obj.board
        sx, sy, ex, ey = move
        moved = board[ey][ex]
        if moved == '.':
            return 0

        enemy = opponent(side)
        enemy_attacks = _pseudo_attacks(board, enemy)
        own_attacks = _pseudo_attacks(board, side)
        moved_value = piece_values[moved[-1]]
        captured_value = piece_values[captured[-1]] if captured != '.' else 0
        bias = 0

        if captured != '.':
            enemy_gain = self._see_gain_on_square(board, ex, ey, enemy, depth=8)
            see_gain = captured_value - enemy_gain
            if see_gain >= 0:
                bias += min(180, captured_value * 0.04 + see_gain * 0.20)
            else:
                bias += max(-240, see_gain * 0.35)
        else:
            enemy_gain = self._see_gain_on_square(board, ex, ey, enemy, depth=6)
            if is_in_check(board, enemy):
                enemy_gain *= 0.55
            if enemy_gain > 0:
                bias -= min(160, enemy_gain * 0.28)

        if (ex, ey) in enemy_attacks and (ex, ey) not in own_attacks:
            bias -= min(90, moved_value * 0.08)

        if moved[-1] == 'z':
            bias += int(_pawn_move_shape_delta(
                side, sx, sy, ex, ey,
                is_capture=(captured != '.'),
                gives_check=is_in_check(board, enemy),
            ) * 0.55)

        return int(bias if side == 'b' else -bias)

    def _rescue_order_score(self, board_obj, move, side):
        board = board_obj.board
        sx, sy, ex, ey = move
        piece = board[sy][sx]
        if piece == '.' or piece[0] != side or piece[-1] == 'j':
            return 0
        value = piece_values[piece[-1]]
        if value < 350:
            return 0
        enemy = opponent(side)
        if not is_attacked(board, sx, sy, enemy):
            return 0

        captured = board_obj.move(sx, sy, ex, ey)
        try:
            if is_in_check(board_obj.board, side):
                return -9000
            landing_attacked = is_attacked(board_obj.board, ex, ey, enemy)
            enemy_gain = self._see_gain_on_square(
                board_obj.board, ex, ey, enemy, depth=6)
            captured_value = piece_values[captured[-1]] if captured != '.' else 0
            if not landing_attacked and enemy_gain <= captured_value + 80:
                return int(14000 + value * 18 + captured_value * 6)
            if enemy_gain > captured_value:
                return -int(min(14000, (enemy_gain - captured_value) * 18))
            return 2500
        finally:
            board_obj.undo_move(sx, sy, ex, ey, captured)

    def _root_light_safety_bias(self, board_obj, move, side, captured):
        board = board_obj.board
        sx, sy, ex, ey = move
        moved = board[ey][ex]
        if moved == '.':
            return 0

        enemy = opponent(side)
        captured_value = piece_values[captured[-1]] if captured != '.' else 0
        enemy_gain = self._see_gain_on_square(board, ex, ey, enemy, depth=5)
        bias = 0

        if captured != '.':
            net = captured_value - enemy_gain
            bias += max(-220, min(150, net * 0.28))
        elif enemy_gain > 0:
            bias -= min(200, enemy_gain * 0.32)

        opening_phase = _non_king_material(board) >= 4200
        if opening_phase:
            move_span = abs(ex - sx) + abs(ey - sy)
            if captured != '.' and moved[-1] != 'j':
                moved_value = piece_values[moved[-1]]
                is_forceful = is_in_check(board, enemy)
                if not is_forceful and captured_value <= moved_value + 120:
                    bias -= 360
                    if moved[-1] == 'p' or move_span >= 5:
                        bias -= 260
                    if enemy_gain > 0:
                        bias -= min(220, enemy_gain * 0.35)

        return int(bias if side == 'b' else -bias)

    def _root_side_score(self, board, side):
        score = evaluate_minimax_static(board)
        return score if side == 'b' else -score

    def _root_reply_safety_score(self, board_obj, move, side):
        board = board_obj.board
        sx, sy, ex, ey = move
        moved = board[sy][sx]
        target = board[ey][ex]
        if moved == '.':
            return -MATE_SCORE

        enemy = opponent(side)
        captured_value = piece_values[target[-1]] if target != '.' else 0
        see = self._see(board_obj, move) if target != '.' else 0

        captured = board_obj.move(sx, sy, ex, ey)
        try:
            terminal = terminal_board_score(board_obj.board)
            if terminal is not None:
                return terminal if side == 'b' else -terminal
            repeat_penalty = self._repetition_penalty_from_count(
                self._repeat_count_current(board_obj.board, enemy))
            enemy_moves = self._get_all_moves(board_obj, enemy)
            if not enemy_moves:
                return MATE_SCORE

            score = self._root_side_score(board_obj.board, side) - repeat_penalty
            if target != '.':
                score += max(-260, min(220, see * 0.45))
            if is_in_check(board_obj.board, enemy):
                score += 90

            enemy_gain = self._see_gain_on_square(board_obj.board, ex, ey, enemy, depth=8)
            if enemy_gain > 0:
                if target != '.':
                    exchange_net = captured_value - enemy_gain
                    score += max(-360, min(130, exchange_net * 0.55))
                else:
                    score -= min(340, enemy_gain * 0.48)

            worst_reply = score
            checked_replies = 0
            for reply in enemy_moves:
                if self._time_up():
                    break
                rsx, rsy, rex, rey = reply
                reply_target = board_obj.board[rey][rex]
                tactical_reply = reply_target != '.' or (rex == ex and rey == ey)
                if not tactical_reply and checked_replies < 5:
                    tactical_reply = self._move_gives_check(board_obj, reply, enemy)
                    checked_replies += 1
                if not tactical_reply:
                    continue

                reply_capture = board_obj.move(rsx, rsy, rex, rey)
                try:
                    our_moves = self._get_all_moves(board_obj, side)
                    if not our_moves:
                        reply_score = -MATE_SCORE
                    else:
                        reply_score = self._root_side_score(board_obj.board, side)
                        if is_in_check(board_obj.board, side):
                            reply_score -= 85
                    worst_reply = min(worst_reply, reply_score)
                finally:
                    board_obj.undo_move(rsx, rsy, rex, rey, reply_capture)

            return worst_reply
        finally:
            board_obj.undo_move(sx, sy, ex, ey, captured)

    def _root_immediate_loss(self, board_obj, move, side):
        sx, sy, ex, ey = move
        enemy = opponent(side)
        captured = board_obj.move(sx, sy, ex, ey)
        try:
            if captured_king_score(captured) is not None:
                return 0
            worst = 0
            board = board_obj.board
            for reply in self._get_all_moves(board_obj, enemy):
                rsx, rsy, rex, rey = reply
                target = board[rey][rex]
                if target == '.' or target[0] != side or target[-1] == 'j':
                    continue
                target_value = piece_values[target[-1]]
                see = self._see(board_obj, reply)
                if see >= -80:
                    loss = max(target_value * 0.75, see)
                else:
                    loss = max(0, see)
                if loss > worst:
                    worst = loss
            return worst
        finally:
            board_obj.undo_move(sx, sy, ex, ey, captured)

    def _root_capture_value(self, board_obj, move):
        sx, sy, ex, ey = move
        target = board_obj.board[ey][ex]
        return piece_values[target[-1]] if target != '.' else 0

    def _root_immediate_net_risk(self, board_obj, move, side):
        return (
            self._root_immediate_loss(board_obj, move, side) -
            self._root_capture_value(board_obj, move)
        )

    def _guard_root_choice(self, board_obj, moves, side, best_move):
        if best_move is None or self._time_up():
            return best_move

        ordered_moves = list(moves)
        self._sort_root_moves(board_obj, ordered_moves, side, best_move)
        candidates = []
        candidate_limit = 28 if self.time_limit >= 2.0 else 20
        candidate_cap = 52 if self.time_limit >= 2.0 else 36
        for move in ordered_moves:
            sx, sy, ex, ey = move
            tactical = board_obj.board[ey][ex] != '.'
            rescue = self._rescue_order_score(board_obj, move, side) > 0
            if (not tactical and len(candidates) >= candidate_limit and
                    len(candidates) < candidate_cap and not rescue):
                tactical = self._move_gives_check(board_obj, move, side)
            if (move == best_move or len(candidates) < candidate_limit or
                    tactical or rescue):
                candidates.append(move)
            if len(candidates) >= candidate_cap and not (tactical or rescue):
                continue
        if best_move not in candidates:
            candidates.insert(0, best_move)

        safety_scores = {}
        for move in candidates:
            if self._time_up():
                break
            safety_scores[move] = self._root_reply_safety_score(board_obj, move, side)
        if best_move not in safety_scores or len(safety_scores) < 2:
            return best_move

        risk_scores = {}
        best_risk = self._root_immediate_net_risk(board_obj, best_move, side)
        if best_risk >= 220:
            for move in candidates:
                if self._time_up():
                    break
                risk_scores[move] = self._root_immediate_net_risk(
                    board_obj, move, side)
            if best_move in risk_scores:
                safest_material_move, safest_risk = min(
                    risk_scores.items(), key=lambda item: item[1])
                if safest_material_move != best_move:
                    safety_gap = (
                        safety_scores.get(safest_material_move, -MATE_SCORE) -
                        safety_scores[best_move]
                    )
                    risk_reduction = best_risk - safest_risk
                    strong_reduction = max(120, best_risk * 0.35)
                    if (risk_reduction >= strong_reduction and
                            safety_gap > -260 and
                            safest_risk <= max(120, best_risk * 0.65)):
                        return safest_material_move

        safest_move, safest_score = max(
            safety_scores.items(), key=lambda item: item[1])
        chosen_score = safety_scores[best_move]
        if safest_move == best_move:
            return best_move

        gap = safest_score - chosen_score
        gives_check = self._move_gives_check(board_obj, best_move, side)
        if chosen_score < -420 and gap > 120:
            return safest_move
        if (not gives_check) and chosen_score < -160 and gap > 320:
            return safest_move
        return best_move

    def _opening_book_move(self, board_obj, side):
        board = board_obj.board
        if side == 'r':
            red_home = (
                board[9] == ['r_c', 'r_m', 'r_x', 'r_s', 'r_j',
                             'r_s', 'r_x', 'r_m', 'r_c'] and
                board[7][1] == 'r_p' and board[7][7] == 'r_p'
            )
            if not red_home:
                return None
            candidates = [
                (7, 7, 4, 7), (1, 7, 4, 7),
                (1, 9, 2, 7), (7, 9, 6, 7),
            ]
            for sx, sy, ex, ey in candidates:
                if (board[sy][sx].startswith(side) and
                        (ex, ey) in get_legal_moves(board_obj, sx, sy)):
                    return (sx, sy, ex, ey)
            return None

        if side != 'b':
            return None
        black_home = (
            board[0] == ['b_c', 'b_m', 'b_x', 'b_s', 'b_j',
                         'b_s', 'b_x', 'b_m', 'b_c'] and
            board[2][1] == 'b_p' and board[2][7] == 'b_p'
        )
        if not black_home:
            return None

        if board[7][4] == 'r_p' and board[7][7] == '.':
            candidates = [(7, 0, 6, 2), (1, 0, 2, 2), (7, 2, 4, 2)]
        elif board[7][4] == 'r_p' and board[7][1] == '.':
            candidates = [(1, 0, 2, 2), (7, 0, 6, 2), (1, 2, 4, 2)]
        else:
            candidates = [
                (1, 0, 2, 2), (7, 0, 6, 2),
                (1, 2, 4, 2), (7, 2, 4, 2),
            ]

        for sx, sy, ex, ey in candidates:
            if board[sy][sx].startswith(side) and (ex, ey) in get_legal_moves(board_obj, sx, sy):
                return (sx, sy, ex, ey)
        return None

    def _development_book_move(self, board_obj, side):
        board = board_obj.board
        if _non_king_material(board) < 4200:
            return None
        if side == 'r':
            candidates = [
                (1, 9, 2, 7), (7, 9, 6, 7),
                (1, 7, 4, 7), (7, 7, 4, 7),
            ]
        else:
            candidates = [
                (1, 0, 2, 2), (7, 0, 6, 2),
                (1, 2, 4, 2), (7, 2, 4, 2),
            ]
        for sx, sy, ex, ey in candidates:
            if (board[sy][sx] != '.' and board[sy][sx][0] == side and
                    (ex, ey) in get_legal_moves(board_obj, sx, sy)):
                captured = board_obj.move(sx, sy, ex, ey)
                try:
                    if not is_in_check(board_obj.board, side):
                        return (sx, sy, ex, ey)
                finally:
                    board_obj.undo_move(sx, sy, ex, ey, captured)
        return None

    def _is_bad_opening_capture(self, board_obj, move, side):
        if move is None:
            return False
        board = board_obj.board
        if _non_king_material(board) < 4200:
            return False
        sx, sy, ex, ey = move
        piece = board[sy][sx]
        target = board[ey][ex]
        if piece == '.' or target == '.' or target[-1] == 'j':
            return False
        if self._move_gives_check(board_obj, move, side):
            return False
        moved_value = piece_values[piece[-1]]
        captured_value = piece_values[target[-1]]
        span = abs(ex - sx) + abs(ey - sy)
        if captured_value > moved_value + 180:
            return False
        return piece[-1] == 'p' or span >= 5

    def _obvious_capture_move(self, board_obj, side, moves=None):
        board = board_obj.board
        best_move = None
        best_score = -float('inf')
        for move in (list(moves) if moves is not None else self._get_all_moves(board_obj, side)):
            sx, sy, ex, ey = move
            target = board[ey][ex]
            if target == '.':
                continue
            if target[-1] == 'j':
                return move

            target_value = piece_values[target[-1]]
            see = self._see(board_obj, move)
            gives_check = self._move_gives_check(board_obj, move, side)
            if target_value < 200 and see < target_value and not gives_check:
                continue
            if target_value >= 400 and see < 180:
                continue
            if see < 260 and not (gives_check and see >= 160):
                continue

            score = see + target_value * 0.18 + (110 if gives_check else 0)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_score >= 360 else None

    def _sort_moves(self, board_obj, moves, ply, tt_move=None,
                    use_neural=False, side=None):
        board = board_obj.board
        killers = self.killers[ply] if ply < len(self.killers) else (None, None)
        priors = {}
        if use_neural and side is not None:
            priors, _ = self._neural_policy_value(board_obj, moves, side)

        def score_move(move):
            score = move_priority(board, move, killers)
            if move == tt_move:
                score += 1000000
            if move in killers:
                score += 9000
            if priors:
                score += int(NEURAL_POLICY_WEIGHT *
                             math.sqrt(max(0.0, priors.get(move, 0.0))))
            score += self.history.get(move, 0)
            sx, sy, ex, ey = move
            target = board[ey][ex]
            if target != '.':
                score += 30000 + self._capture_order_score(
                    board_obj, move, force_see=(ply <= 1))
            should_check = ply <= 2 or target != '.' or move == tt_move
            if should_check and self._move_gives_check(board_obj, move, board[sy][sx][0]):
                score += 18000 if target == '.' else 9000
            return score

        moves.sort(key=score_move, reverse=True)

    def _root_move_score(self, board_obj, move, side, preferred, priors=None):
        board = board_obj.board
        sx, sy, ex, ey = move
        score = move_priority(board, move, self.killers[0])
        score += self.history.get(move, 0)
        if move == preferred:
            score += 1000000
        if priors:
            score += int(NEURAL_POLICY_WEIGHT *
                         math.sqrt(max(0.0, priors.get(move, 0.0))))

        target = board[ey][ex]
        if target != '.':
            see = self._see(board_obj, move)
            score += 45000 + max(-35000, min(35000, see * 32))
            if see < 0:
                score -= min(26000, -see * 24)
        if target == '.' and board[sy][sx][-1] == 'j':
            score -= 20000

        gives_check = self._move_gives_check(board_obj, move, side)
        if gives_check:
            score += 26000 if target == '.' else 22000
        score += _pawn_move_shape_score(
            board, move, side, gives_check=gives_check) * 35
        score -= self._root_repetition_penalty(board_obj, move, side)
        return score

    def _sort_root_moves(self, board_obj, moves, side, preferred):
        priors, _ = self._neural_policy_value(board_obj, moves, side)
        moves.sort(
            key=lambda move: self._root_move_score(
                board_obj, move, side, preferred, priors),
            reverse=True
        )

    def _root_search_moves(self, board_obj, moves, side, depth, preferred):
        if depth < 2:
            return moves
        if depth == 2:
            if self.time_limit >= 8.0:
                soft_limit, hard_limit = 20, 30
            elif self.time_limit >= 4.0:
                soft_limit, hard_limit = 14, 22
            else:
                soft_limit, hard_limit = 10, 16
        elif depth == 3:
            if self.time_limit >= 8.0:
                soft_limit, hard_limit = 12, 18
            elif self.time_limit >= 4.0:
                soft_limit, hard_limit = 8, 12
            else:
                soft_limit, hard_limit = 6, 10
        else:
            soft_limit = 14 if self.time_limit >= 4.5 else 10
            hard_limit = 22 if self.time_limit >= 4.5 else 16
        selected = []
        seen = set()
        for move in moves:
            sx, sy, ex, ey = move
            tactical = (
                move == preferred or
                board_obj.board[ey][ex] != '.' or
                self._move_gives_check(board_obj, move, side) or
                self._rescue_order_score(board_obj, move, side) > 0
            )
            if len(selected) < soft_limit or tactical:
                selected.append(move)
                seen.add(move)
            if len(selected) >= hard_limit:
                break
        if preferred is not None and preferred not in seen and preferred in moves:
            selected.append(preferred)
        return selected or moves

    def _find_tactical_finish(self, board_obj, side, moves=None):
        enemy = opponent(side)
        moves = list(moves) if moves is not None else self._get_all_moves(board_obj, side)
        self._sort_moves(board_obj, moves, 0)
        for sx, sy, ex, ey in moves:
            target = board_obj.board[ey][ex]
            if target == enemy + '_j':
                return (sx, sy, ex, ey)
        for sx, sy, ex, ey in moves:
            captured = board_obj.move(sx, sy, ex, ey)
            gives_check = is_in_check(board_obj.board, enemy)
            mate = gives_check and not self._get_all_moves(board_obj, enemy)
            board_obj.undo_move(sx, sy, ex, ey, captured)
            if mate:
                return (sx, sy, ex, ey)
        return None

    def _mate_in_one_move(self, board_obj, side, moves=None):
        enemy = opponent(side)
        moves = list(moves) if moves is not None else self._get_all_moves(board_obj, side)
        self._sort_moves(board_obj, moves, 0)
        for sx, sy, ex, ey in moves:
            target = board_obj.board[ey][ex]
            if target == enemy + '_j':
                return (sx, sy, ex, ey)

            captured = board_obj.move(sx, sy, ex, ey)
            try:
                if (is_in_check(board_obj.board, enemy) and
                        not self._get_all_moves(board_obj, enemy)):
                    return (sx, sy, ex, ey)
            finally:
                board_obj.undo_move(sx, sy, ex, ey, captured)
        return None

    def _find_mate_threat_defense(self, board_obj, side, moves=None):
        enemy = opponent(side)
        if self._mate_in_one_move(board_obj, enemy) is None:
            return None

        best_move = None
        best_score = -float('inf')
        moves = list(moves) if moves is not None else self._get_all_moves(board_obj, side)
        self._sort_moves(board_obj, moves, 0)
        for sx, sy, ex, ey in moves:
            captured = board_obj.move(sx, sy, ex, ey)
            try:
                if self._mate_in_one_move(board_obj, enemy) is not None:
                    continue
                score = self._root_side_score(board_obj.board, side)
                if captured != '.':
                    score += piece_values[captured[-1]] * 0.20
                if is_in_check(board_obj.board, enemy):
                    score += 120
                if score > best_score:
                    best_score = score
                    best_move = (sx, sy, ex, ey)
            finally:
                board_obj.undo_move(sx, sy, ex, ey, captured)
        return best_move

    # ── 静态搜索（Quiescence Search）────────────────────────────────────────
    #
    #  在叶节点继续搜索所有吃子着法，直到局面"平静"（无子可吃），
    #  彻底消除"视野边界"错觉：AI 不会因为深度截断而看不到下一步的反吃。

    def _neural_policy_value(self, board_obj, moves, side):
        advisor = self._ensure_neural()
        if advisor is None or not moves or self._time_up():
            return {}, None
        return advisor.policy_value(board_obj.board, moves, side)

    def _neural_value_bias(self, board, side, weight):
        advisor = self._ensure_neural()
        if advisor is None or self._time_up():
            return 0
        value = advisor.value(board, side)
        if value is None:
            return 0
        return int(value * weight)

    def _quiesce(self, board_obj, alpha, beta, maximizing, checks_left=1):
        self.nodes += 1
        if self._time_up():
            self._timeout = True
            return self._eval(board_obj.board)
        board = board_obj.board
        side = 'b' if maximizing else 'r'

        if is_in_check(board, side):
            moves = self._get_all_moves(board_obj, side)
            if not moves:
                return -MATE_SCORE if maximizing else MATE_SCORE
            self._sort_moves(board_obj, moves, 0)
            if maximizing:
                value = -float('inf')
                for sx, sy, ex, ey in moves:
                    captured = board_obj.move(sx, sy, ex, ey)
                    terminal = captured_king_score(captured)
                    if terminal is None:
                        terminal = self._quiesce(board_obj, alpha, beta, False,
                                                 checks_left)
                    value = max(value, terminal)
                    board_obj.undo_move(sx, sy, ex, ey, captured)
                    alpha = max(alpha, value)
                    if alpha >= beta or self._timeout:
                        break
                return value
            value = float('inf')
            for sx, sy, ex, ey in moves:
                captured = board_obj.move(sx, sy, ex, ey)
                terminal = captured_king_score(captured)
                if terminal is None:
                    terminal = self._quiesce(board_obj, alpha, beta, True,
                                             checks_left)
                value = min(value, terminal)
                board_obj.undo_move(sx, sy, ex, ey, captured)
                beta = min(beta, value)
                if alpha >= beta or self._timeout:
                    break
            return value

        # 静态评估作为"不吃子"基准分
        stand_pat = self._eval(board)
        if maximizing:
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if stand_pat < beta:
                beta = stand_pat

        for sx, sy, ex, ey in self._get_noisy_moves(board_obj, side, checks_left):
            is_quiet_check = board[ey][ex] == '.'
            captured = board_obj.move(sx, sy, ex, ey)
            next_checks = checks_left - 1 if is_quiet_check else checks_left
            score = captured_king_score(captured)
            if score is None:
                score = self._quiesce(board_obj, alpha, beta, not maximizing,
                                      next_checks)
            board_obj.undo_move(sx, sy, ex, ey, captured)

            if maximizing:
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            else:
                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score

        return alpha if maximizing else beta

    # ── Alpha-Beta 主搜索 ─────────────────────────────────────────────────────

    def _search(self, board_obj, depth, alpha, beta, maximizing, ply, allow_null=True):
        self.nodes += 1

        if self._timeout or self._time_up():
            self._timeout = True
            return 0

        board = board_obj.board
        zh = compute_hash(board)
        tt_key = (zh, maximizing)
        tt_move = None
        side = 'b' if maximizing else 'r'

        # 查置换表
        if tt_key in self.tt:
            cached_depth, cached_score, flag, cached_move = self.tt[tt_key]
            tt_move = cached_move
            if cached_depth >= depth:
                if flag == TT_EXACT:
                    return cached_score
                elif flag == TT_LOWER:
                    alpha = max(alpha, cached_score)
                elif flag == TT_UPPER:
                    beta = min(beta, cached_score)
                if alpha >= beta:
                    return cached_score

        # 到达叶节点：进入静态搜索
        if depth == 0:
            if ply >= 6 and not is_in_check(board, side):
                return self._eval(board, side, allow_neural=True)
            if self.time_limit >= 8.0 and ply <= 4:
                checks_left = 1
            elif self.time_limit >= 4.0 and ply <= 2:
                checks_left = 1
            else:
                checks_left = 0
            return self._quiesce(board_obj, alpha, beta, maximizing, checks_left)

        in_check = is_in_check(board, side)

        if (allow_null and depth >= 4 and not in_check and
                _non_king_material(board) >= 1400 and
                _side_non_king_material(board, side) >= 500):
            reduction = 2 if depth < 5 else 3
            null_score = self._search(
                board_obj, depth - 1 - reduction,
                alpha, beta, not maximizing, ply + 1, allow_null=False)
            if maximizing and null_score >= beta:
                return beta
            if (not maximizing) and null_score <= alpha:
                return alpha

        moves = self._get_all_moves(board_obj, side)

        if not moves:
            if is_in_check(board, side):
                return (-MATE_SCORE + ply) if maximizing else (MATE_SCORE - ply)
            return 0

        # 着法排序
        self._sort_moves(
            board_obj, moves, ply, tt_move,
            use_neural=False,
            side=side,
        )

        best_score = -float('inf') if maximizing else float('inf')
        best_move = moves[0]
        orig_alpha = alpha
        orig_beta = beta

        for move_index, (sx, sy, ex, ey) in enumerate(moves):
            if self._timeout:
                break

            is_capture = board[ey][ex] != '.'
            capture_see = self._see(board_obj, (sx, sy, ex, ey)) if is_capture else 0
            captured = board_obj.move(sx, sy, ex, ey)
            score = captured_king_score(captured, ply)
            if score is None:
                next_side = opponent(side)
                gives_check = is_in_check(board_obj.board, next_side)
                quiet_limit = 36 if depth == 2 else 24
                if (depth <= 2 and move_index >= quiet_limit and
                        not is_capture and not gives_check and not in_check):
                    board_obj.undo_move(sx, sy, ex, ey, captured)
                    continue
                target_value = piece_values[captured[-1]] if is_capture else 0
                tactical_capture = is_capture and (
                    target_value >= 400 or capture_see >= 260)
                extension = 1 if (
                    depth <= 3 and ply < 8 and
                    (gives_check or tactical_capture)
                ) else 0
                full_depth = depth - 1 + extension
                reduced = (
                    depth >= 3 and move_index >= 6 and
                    not is_capture and not gives_check and not in_check
                )
                if reduced:
                    score = self._search(board_obj, max(0, full_depth - 1),
                                         alpha, beta, not maximizing, ply + 1)
                    needs_research = (
                        (maximizing and score > alpha) or
                        ((not maximizing) and score < beta)
                    )
                    if needs_research and not self._timeout:
                        score = self._search(board_obj, full_depth,
                                             alpha, beta, not maximizing, ply + 1)
                elif move_index > 0 and maximizing and alpha != -float('inf'):
                    score = self._search(board_obj, full_depth,
                                         alpha, alpha + 1, not maximizing, ply + 1)
                    if alpha < score < beta and not self._timeout:
                        score = self._search(board_obj, full_depth,
                                             alpha, beta, not maximizing, ply + 1)
                elif move_index > 0 and (not maximizing) and beta != float('inf'):
                    score = self._search(board_obj, full_depth,
                                         beta - 1, beta, not maximizing, ply + 1)
                    if alpha < score < beta and not self._timeout:
                        score = self._search(board_obj, full_depth,
                                             alpha, beta, not maximizing, ply + 1)
                else:
                    score = self._search(board_obj, full_depth,
                                         alpha, beta, not maximizing, ply + 1)
            board_obj.undo_move(sx, sy, ex, ey, captured)

            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = (sx, sy, ex, ey)
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = (sx, sy, ex, ey)
                beta = min(beta, best_score)

            if alpha >= beta:
                # 记录杀手着法（只记非吃子）
                if not is_capture and ply < len(self.killers):
                    km = (sx, sy, ex, ey)
                    if km != self.killers[ply][0]:
                        self.killers[ply][1] = self.killers[ply][0]
                        self.killers[ply][0] = km
                # 历史启发加分
                cutoff_move = (sx, sy, ex, ey)
                self.history[cutoff_move] = self.history.get(cutoff_move, 0) + depth * depth
                break

        # 写置换表
        if not self._timeout:
            if best_score <= orig_alpha:
                flag = TT_UPPER
            elif best_score >= orig_beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            old = self.tt.get(tt_key)
            if old is None or depth >= old[0]:
                self.tt[tt_key] = (depth, best_score, flag, best_move)

        return best_score

    # ── 迭代加深入口 ──────────────────────────────────────────────────────────

    def _search_root(self, board_obj, moves, side, depth,
                     alpha, beta, preferred_move):
        maximizing = (side == 'b')
        self._sort_root_moves(board_obj, moves, side, preferred_move)
        search_moves = self._root_search_moves(
            board_obj, moves, side, depth, preferred_move)
        self.last_root_searched = 0

        iter_best_score = -float('inf') if maximizing else float('inf')
        iter_best_move = preferred_move
        root_alpha = alpha
        root_beta = beta

        for move_index, (sx, sy, ex, ey) in enumerate(search_moves):
            if self._time_up():
                self._timeout = True
                break

            captured = board_obj.move(sx, sy, ex, ey)
            score = captured_king_score(captured)
            if score is None:
                score = self._search(
                    board_obj, depth - 1,
                    root_alpha, root_beta,
                    not maximizing, ply=1
                )
            if not self._timeout:
                score += self._root_light_safety_bias(
                    board_obj, (sx, sy, ex, ey), side, captured)
                score += self._root_tactical_bias(
                    board_obj, (sx, sy, ex, ey), side, captured)
                if captured != '.':
                    score += self._neural_value_bias(
                        board_obj.board, opponent(side),
                        NEURAL_ROOT_VALUE_WEIGHT)
                repeat_penalty = self._repetition_penalty_from_count(
                    self._repeat_count_current(board_obj.board, opponent(side)))
                if repeat_penalty:
                    score += -repeat_penalty if side == 'b' else repeat_penalty
            board_obj.undo_move(sx, sy, ex, ey, captured)
            if self._timeout:
                break
            self.last_root_searched += 1

            if maximizing:
                if score > iter_best_score:
                    iter_best_score = score
                    iter_best_move = (sx, sy, ex, ey)
                root_alpha = max(root_alpha, iter_best_score)
            else:
                if score < iter_best_score:
                    iter_best_score = score
                    iter_best_move = (sx, sy, ex, ey)
                root_beta = min(root_beta, iter_best_score)

            if root_alpha >= root_beta:
                break

        return iter_best_score, iter_best_move

    def get_move(self, board_obj, side):
        self._reset()
        maximizing = (side == 'b')

        all_moves = self._get_all_moves(board_obj, side)
        if not all_moves:
            return None
        if len(all_moves) == 1:
            return all_moves[0]
        moves = self._filter_repeating_root_moves(board_obj, all_moves, side)

        tactical = self._find_tactical_finish(board_obj, side, moves)
        if tactical is not None:
            return tactical

        defense = self._find_mate_threat_defense(board_obj, side, moves)
        if defense is not None:
            return defense

        book_move = self._opening_book_move(board_obj, side)
        strict_book = book_move is not None
        if book_move is None:
            dev_move = self._development_book_move(board_obj, side)
            if (dev_move is not None and
                    self._root_immediate_net_risk(board_obj, dev_move, side) <= 180):
                book_move = dev_move
        if book_move not in moves:
            book_move = None
            strict_book = False

        obvious_capture = self._obvious_capture_move(board_obj, side, moves)
        if obvious_capture is not None and self.time_limit < 0.35:
            return self._guard_root_choice(board_obj, moves, side, obvious_capture)

        if self.use_neural and self.time_limit >= 0.8:
            self.neural = get_neural_advisor()
            if self.neural is not None:
                if self.time_limit >= 4.0:
                    self.neural_leaf_budget = 96
                elif self.time_limit >= 2.0:
                    self.neural_leaf_budget = 48
                else:
                    self.neural_leaf_budget = 16
                self._neural_policy_value(board_obj, moves, side)

        start = time.time()
        if self.time_limit >= 4.0:
            guard_reserve = 0.12
        elif self.time_limit >= 1.0:
            guard_reserve = 0.18
        else:
            guard_reserve = max(0.08, self.time_limit * 0.35)
        search_budget = max(0.04, self.time_limit - guard_reserve)
        self.deadline = start + search_budget
        best_move = book_move or obvious_capture or moves[0]
        last_score = None

        for depth in range(1, self.max_depth + 1):
            if self._time_up():
                break

            self._timeout = False

            use_aspiration = (
                last_score is not None and
                depth >= 4 and
                self.time_limit >= 12.0
            )
            if use_aspiration:
                window = 90 + depth * 20
                alpha = last_score - window
                beta = last_score + window
                iter_best_score, iter_best_move = self._search_root(
                    board_obj, moves, side, depth, alpha, beta, best_move)

                if (not self._timeout and
                        (iter_best_score <= alpha or iter_best_score >= beta)):
                    self._timeout = False
                    alpha = last_score - window * 4
                    beta = last_score + window * 4
                    iter_best_score, iter_best_move = self._search_root(
                        board_obj, moves, side, depth, alpha, beta, best_move)

                if (not self._timeout and
                        (iter_best_score <= alpha or iter_best_score >= beta)):
                    self._timeout = False
                    iter_best_score, iter_best_move = self._search_root(
                        board_obj, moves, side, depth,
                        -float('inf'), float('inf'), best_move)
            else:
                iter_best_score, iter_best_move = self._search_root(
                    board_obj, moves, side, depth,
                    -float('inf'), float('inf'), best_move)

            if not self._timeout:
                best_move = iter_best_move
                last_score = iter_best_score
                self.completed_depth = depth
            else:
                if (depth >= 3 and iter_best_move is not None and
                        self.last_root_searched >= 3):
                    best_move = iter_best_move
                    last_score = iter_best_score
                    self.partial_depth = depth
                break

        guard_budget = max(0.08, guard_reserve)
        self._timeout = False
        self.deadline = time.time() + guard_budget
        final_move = self._guard_root_choice(board_obj, moves, side, best_move)
        if strict_book and book_move is not None:
            final_risk = self._root_immediate_net_risk(board_obj, final_move, side)
            book_risk = self._root_immediate_net_risk(board_obj, book_move, side)
            if self._is_bad_opening_capture(board_obj, final_move, side):
                if final_risk > -250 and book_risk <= final_risk + 350:
                    return book_move
                if book_risk + 120 < final_risk:
                    return book_move
            fsx, fsy, fex, fey = final_move
            quiet_nonforcing = (
                board_obj.board[fey][fex] == '.' and
                not self._move_gives_check(board_obj, final_move, side)
            )
            if (final_move != book_move and quiet_nonforcing and
                    book_risk <= final_risk + 80):
                return book_move
        return final_move


# ─────────────────────────────────────────────────────────────────────────────
#  蒙特卡洛树搜索（MCTS）
# ─────────────────────────────────────────────────────────────────────────────

def _mcts_move_score(board_obj, move, side):
    board = board_obj.board
    sx, sy, ex, ey = move
    piece = board[sy][sx]
    target = board[ey][ex]
    score = random.random() * 0.01
    if target != '.':
        score += piece_values[target[-1]] * 12 - piece_values[piece[-1]]
    if piece[-1] == 'z':
        if (side == 'b' and ey > sy) or (side == 'r' and ey < sy):
            score += 25
    captured = board_obj.move(sx, sy, ex, ey)
    try:
        if is_in_check(board_obj.board, opponent(side)):
            score += 450
        score += evaluate_fast(board_obj.board) * (0.010 if side == 'b' else -0.010)
    finally:
        board_obj.undo_move(sx, sy, ex, ey, captured)
    return score


def _mcts_fast_move_score(board, move, side):
    sx, sy, ex, ey = move
    piece = board[sy][sx]
    target = board[ey][ex]
    if piece == '.':
        return -100000
    score = random.random() * 0.01
    if target != '.':
        score += piece_values[target[-1]] * 9 - piece_values[piece[-1]] * 0.7
    ptype = piece[-1]
    if ptype == 'z':
        if (side == 'b' and ey > sy) or (side == 'r' and ey < sy):
            score += 28
        if 3 <= ex <= 5:
            score += 10
    elif ptype in ('m', 'c', 'p') and 2 <= ex <= 6 and 2 <= ey <= 7:
        score += 8
    elif ptype == 'j':
        score -= 45
    return score


def _mcts_rollout_move(board_obj, moves, side):
    if len(moves) == 1:
        return moves[0]
    if random.random() < 0.05:
        return random.choice(moves)
    board = board_obj.board
    captures = [m for m in moves if board[m[3]][m[2]] != '.']
    if captures:
        pool = captures
    elif len(moves) > 12:
        pool = random.sample(moves, 12)
    else:
        pool = moves
    ranked = sorted(pool, key=lambda m: _mcts_fast_move_score(board, m, side),
                    reverse=True)
    if len(ranked) >= 2 and random.random() < 0.18:
        return ranked[1]
    return ranked[0]


def _mcts_side_eval(board, side):
    score = evaluate_fast(board)
    return score if side == 'b' else -score


def _mcts_tactical_score(board_obj, move, side, prior=0.0):
    sx, sy, ex, ey = move
    captured = board_obj.move(sx, sy, ex, ey)
    try:
        enemy = opponent(side)
        enemy_moves = get_all_moves(board_obj, enemy)
        if not enemy_moves:
            return MATE_SCORE

        base = _mcts_side_eval(board_obj.board, side)
        if captured != '.':
            base += min(60, piece_values[captured[-1]] * 0.04)
        if is_in_check(board_obj.board, enemy):
            base += 55

        worst_after_reply = base
        board = board_obj.board
        for reply in enemy_moves:
            rsx, rsy, rex, rey = reply
            target = board[rey][rex]
            is_tactical = target != '.'
            if not is_tactical:
                reply_capture = board_obj.move(rsx, rsy, rex, rey)
                try:
                    is_tactical = is_in_check(board_obj.board, side)
                finally:
                    board_obj.undo_move(rsx, rsy, rex, rey, reply_capture)
            if not is_tactical:
                continue

            reply_capture = board_obj.move(rsx, rsy, rex, rey)
            try:
                reply_score = _mcts_side_eval(board_obj.board, side)
                if is_in_check(board_obj.board, side):
                    reply_score -= 45
                worst_after_reply = min(worst_after_reply, reply_score)
            finally:
                board_obj.undo_move(rsx, rsy, rex, rey, reply_capture)

        return worst_after_reply + min(45, prior * 45)
    finally:
        board_obj.undo_move(sx, sy, ex, ey, captured)


def _mcts_guarded_move(board_obj, side, proposed, policy=None):
    legal_moves = get_all_moves(board_obj, side)
    if not legal_moves:
        return None
    policy = policy or {}
    legal_set = set(legal_moves)
    if proposed not in legal_set:
        proposed = None

    best_move = None
    best_score = -float('inf')
    proposed_score = None
    for move in legal_moves:
        score = _mcts_tactical_score(board_obj, move, side, policy.get(move, 0.0))
        if move == proposed:
            proposed_score = score
        if score > best_score:
            best_score = score
            best_move = move

    if proposed is None:
        return best_move
    if proposed_score is None:
        return best_move
    gap = best_score - proposed_score
    if gap > 90 or (proposed_score < 80 and gap > 35):
        return best_move
    return proposed


class MCTSNode:
    __slots__ = ('board_obj', 'side', 'parent', 'move',
                 'children', 'wins', 'visits', 'untried')

    def __init__(self, board_obj, side, parent=None, move=None):
        self.board_obj = board_obj
        self.side = side
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried = get_all_moves(board_obj, side)
        random.shuffle(self.untried)
        self.untried.sort(
            key=lambda m: _mcts_fast_move_score(board_obj.board, m, side))

    def ucb(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        return -self.wins / self.visits + c * math.sqrt(
            math.log(self.parent.visits) / self.visits)

    def best_child(self):
        return max(self.children, key=lambda n: n.ucb())

    def expand(self):
        move = self.untried.pop()
        new_board = copy.deepcopy(self.board_obj)
        new_board.move(*move)
        child = MCTSNode(new_board, 'r' if self.side == 'b' else 'b',
                         parent=self, move=move)
        self.children.append(child)
        return child

    def rollout(self, max_steps=22):
        board = copy.deepcopy(self.board_obj)
        side = self.side
        seen = {}
        for _ in range(max_steps):
            key = (side, tuple(tuple(row) for row in board.board))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 3:
                break
            moves = get_all_moves(board, side)
            if not moves:
                black_result = 1 if side == 'r' else -1
                return black_result if self.side == 'b' else -black_result
            board.move(*_mcts_rollout_move(board, moves, side))
            side = 'r' if side == 'b' else 'b'
        score = evaluate_fast(board.board)
        black_result = 1 if score > 0 else (-1 if score < 0 else 0)
        return black_result if self.side == 'b' else -black_result

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(-result)


class MCTSAI:
    def __init__(self, time_limit=2.0, use_neural=False):
        self.time_limit = time_limit
        self.use_neural = use_neural
        self.last_backend = None
        self.last_simulations = 0
        self.last_raw_move = None
        self.last_guarded = False
        self.last_refined = False
        self._helper = MinimaxAI(max_depth=2, time_limit=0.15, use_neural=False)
        self.repetition_counts = {}
        self.repetition_seen_limit = 2

    def set_repetition_context(self, position_counts=None, seen_limit=2):
        self.repetition_counts = dict(position_counts or {})
        self.repetition_seen_limit = max(1, seen_limit)
        self._helper.set_repetition_context(
            self.repetition_counts, self.repetition_seen_limit)

    def _neural_simulations(self, advisor):
        if advisor is None:
            return 0
        if advisor.device_name == "cuda":
            return max(8, min(64, int(self.time_limit * 8)))
        return max(4, min(24, int(self.time_limit * 4)))

    def _neural_move(self, board_obj, side):
        if not self.use_neural:
            return None
        advisor = get_neural_advisor()
        if advisor is None:
            return None
        simulations = self._neural_simulations(advisor)
        if simulations <= 0:
            return None
        try:
            from alpha_mcts import AlphaMCTS

            mcts = AlphaMCTS(
                advisor.model,
                simulations=simulations,
                c_puct=2.1,
                device=advisor.device_name,
            )
            move, policy = mcts.get_move(
                board_obj, side, temperature=0, add_noise=False)
            guarded = _mcts_guarded_move(board_obj, side, move, policy)
            self.last_backend = "neural"
            self.last_simulations = simulations
            self.last_raw_move = move
            self.last_guarded = guarded != move
            return guarded
        except Exception as exc:
            print(f"[MCTS] neural backend failed: {exc}")
            return None

    def _opening_book_move(self, board_obj, side):
        helper = self._helper
        helper._reset()
        move = helper._opening_book_move(board_obj, side)
        if move is not None:
            return move
        return helper._development_book_move(board_obj, side)

    def _urgent_tactical_move(self, board_obj, side):
        helper = self._helper
        helper._reset()
        move = helper._find_tactical_finish(board_obj, side)
        if move is not None:
            return move
        return helper._find_mate_threat_defense(board_obj, side)

    def _refine_with_minimax(self, board_obj, side, mcts_move, policy, budget):
        if budget < 0.18 or mcts_move is None:
            return mcts_move
        try:
            depth = 5 if self.time_limit >= 3.0 else 4
            tactical_ai = MinimaxAI(
                max_depth=depth,
                time_limit=budget,
                use_neural=False,
            )
            tactical_ai.set_repetition_context(
                self.repetition_counts, self.repetition_seen_limit)
            mm_move = tactical_ai.get_move(board_obj, side)
            if mm_move is None or mm_move == mcts_move:
                return mcts_move

            mcts_score = _mcts_tactical_score(
                board_obj, mcts_move, side, policy.get(mcts_move, 0.0))
            mm_score = _mcts_tactical_score(
                board_obj, mm_move, side, policy.get(mm_move, 0.0))

            slack = 190 if tactical_ai.completed_depth >= 2 else 80
            if mm_score >= mcts_score - slack:
                self.last_refined = True
                return mm_move
            if mcts_score < 40 and mm_score >= mcts_score - max(slack, 140):
                self.last_refined = True
                return mm_move
            return mcts_move
        except Exception as exc:
            print(f"[MCTS] minimax refine failed: {exc}")
            return mcts_move

    def _fallback_move(self, board_obj, side, preferred_move=None):
        root = MCTSNode(copy.deepcopy(board_obj), side)
        if not root.untried:
            return None
        if preferred_move in root.untried:
            root.untried.remove(preferred_move)
            root.untried.append(preferred_move)
        start = time.time()
        reserve = min(0.85, max(0.18, self.time_limit * 0.28))
        search_until = start + max(0.05, self.time_limit - reserve)
        while time.time() < search_until:
            node = root
            while not node.untried and node.children:
                node = node.best_child()
            if node.untried:
                node = node.expand()
            result = node.rollout()
            node.backpropagate(result)
        if not root.children:
            moves = get_all_moves(board_obj, side)
            if preferred_move in moves:
                return preferred_move
            return random.choice(moves) if moves else None
        self.last_backend = "heuristic"
        self.last_simulations = root.visits
        total_visits = sum(n.visits for n in root.children) or 1
        policy = {n.move: n.visits / total_visits for n in root.children}

        def final_score(node):
            visits = max(1, node.visits)
            q_root = -node.wins / visits
            tactical = _mcts_tactical_score(
                board_obj, node.move, side, policy.get(node.move, 0.0))
            return (
                tactical
                + q_root * 420
                + math.log1p(visits) * 28
                + (45 if node.move == preferred_move else 0)
            )

        candidates = root.children
        if len(candidates) > 18:
            board = board_obj.board
            candidates = sorted(
                candidates,
                key=lambda n: (
                    n.visits,
                    policy.get(n.move, 0.0),
                    _mcts_fast_move_score(board, n.move, side),
                ),
                reverse=True,
            )[:18]
            if preferred_move is not None:
                preferred_child = next(
                    (n for n in root.children if n.move == preferred_move),
                    None,
                )
                if preferred_child is not None and preferred_child not in candidates:
                    candidates.append(preferred_child)

        raw_move = max(candidates, key=final_score).move
        refine_budget = max(0.0, start + self.time_limit - time.time() - 0.08)
        refined_move = self._refine_with_minimax(
            board_obj, side, raw_move, policy, refine_budget)
        guarded = _mcts_guarded_move(board_obj, side, refined_move, policy)
        self.last_raw_move = raw_move
        self.last_guarded = guarded != refined_move
        if self.last_refined:
            self.last_backend = "hybrid"
        return guarded

    def get_move(self, board_obj, side):
        self.last_backend = None
        self.last_simulations = 0
        self.last_raw_move = None
        self.last_guarded = False
        self.last_refined = False

        urgent_move = self._urgent_tactical_move(board_obj, side)
        if urgent_move is not None:
            self.last_backend = "tactic"
            self.last_raw_move = urgent_move
            return urgent_move

        book_move = self._opening_book_move(board_obj, side)
        if self.use_neural:
            neural_move = self._neural_move(board_obj, side)
            if neural_move is not None:
                return neural_move

        move = self._fallback_move(board_obj, side, preferred_move=book_move)
        if move is not None:
            return move
        if book_move is not None:
            self.last_backend = "book"
            self.last_raw_move = book_move
            return book_move
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  简单 AI（对比用）
# ─────────────────────────────────────────────────────────────────────────────

class StrongAI:
    """Hybrid high-strength mode: deep minimax plus tactical verification."""

    def __init__(self, max_depth=10, time_limit=8.0, use_mcts=True):
        self.max_depth = max(1, max_depth)
        self.time_limit = max(0.1, float(time_limit))
        self.use_mcts = use_mcts
        self.repetition_counts = {}
        self.repetition_seen_limit = 2
        self.last_main_depth = 0
        self.last_main_partial = 0
        self.last_choice_reason = ""
        self.last_candidates = 0
        self.last_mcts_move = None

    def set_repetition_context(self, position_counts=None, seen_limit=2):
        self.repetition_counts = dict(position_counts or {})
        self.repetition_seen_limit = max(1, seen_limit)

    def _new_minimax(self, time_limit, max_depth=None):
        ai = MinimaxAI(
            max_depth=max_depth or self.max_depth,
            time_limit=max(0.05, time_limit),
            use_neural=False,
        )
        ai.set_repetition_context(
            self.repetition_counts, self.repetition_seen_limit)
        return ai

    def _add_candidate(self, move, legal_set, candidates, seen):
        if move is None or move not in legal_set or move in seen:
            return
        candidates.append(move)
        seen.add(move)

    def _probe_candidates(self, board_obj, side, preferred, budget):
        probe = self._new_minimax(budget, max_depth=4)
        probe._reset()
        probe.deadline = time.time() + max(0.05, budget)
        all_moves = probe._get_all_moves(board_obj, side)
        if not all_moves:
            return [], probe
        moves = probe._filter_repeating_root_moves(board_obj, all_moves, side)
        legal_set = set(moves)
        candidates = []
        seen = set()

        self._add_candidate(preferred, legal_set, candidates, seen)
        self._add_candidate(
            probe._find_tactical_finish(board_obj, side, moves),
            legal_set, candidates, seen)
        self._add_candidate(
            probe._find_mate_threat_defense(board_obj, side, moves),
            legal_set, candidates, seen)
        self._add_candidate(
            probe._obvious_capture_move(board_obj, side, moves),
            legal_set, candidates, seen)
        self._add_candidate(
            probe._opening_book_move(board_obj, side),
            legal_set, candidates, seen)
        self._add_candidate(
            probe._development_book_move(board_obj, side),
            legal_set, candidates, seen)

        ordered = list(moves)
        probe._sort_root_moves(board_obj, ordered, side, preferred)
        limit = 22 if self.time_limit >= 8.0 else 16
        for move in ordered:
            if probe._time_up():
                break
            sx, sy, ex, ey = move
            target = board_obj.board[ey][ex]
            rescue = probe._rescue_order_score(board_obj, move, side) > 0
            capture = target != '.'
            check = (not capture or rescue) and probe._move_gives_check(
                board_obj, move, side)
            if (len(candidates) < 8 or rescue or check or
                    (capture and piece_values[target[-1]] >= 200)):
                self._add_candidate(move, legal_set, candidates, seen)
            if len(candidates) >= limit:
                break

        return candidates, probe

    def _mcts_proposal(self, board_obj, side, budget, legal_set):
        self.last_mcts_move = None
        if not self.use_mcts or budget < 0.45:
            return None
        try:
            mcts = MCTSAI(time_limit=budget, use_neural=False)
            mcts.set_repetition_context(
                self.repetition_counts, self.repetition_seen_limit)
            move = mcts.get_move(board_obj, side)
            if move in legal_set:
                self.last_mcts_move = move
                return move
        except Exception as exc:
            print(f"[StrongAI] mcts proposal failed: {exc}")
        return None

    def _verified_line_score(self, verifier, board_obj, move, side, depth):
        sx, sy, ex, ey = move
        captured = board_obj.move(sx, sy, ex, ey)
        try:
            terminal = terminal_board_score(board_obj.board, ply=1)
            if terminal is not None:
                raw = terminal
            else:
                raw = verifier._search(
                    board_obj,
                    max(0, depth),
                    -MATE_SCORE,
                    MATE_SCORE,
                    opponent(side) == 'b',
                    ply=1,
                    allow_null=False,
                )
            return raw if side == 'b' else -raw
        finally:
            board_obj.undo_move(sx, sy, ex, ey, captured)

    def _score_candidate(self, verifier, board_obj, move, side,
                         main_move, mcts_move, line_depth):
        if verifier._time_up():
            return None
        sx, sy, ex, ey = move
        board = board_obj.board
        target = board[ey][ex]
        capture_value = piece_values[target[-1]] if target != '.' else 0

        line = self._verified_line_score(
            verifier, board_obj, move, side, line_depth)
        if verifier._timeout and move != main_move:
            return None

        safety = verifier._root_reply_safety_score(board_obj, move, side)
        risk = verifier._root_immediate_net_risk(board_obj, move, side)
        gives_check = verifier._move_gives_check(board_obj, move, side)
        rescue = verifier._rescue_order_score(board_obj, move, side) > 0

        score = line + safety * 0.30
        if risk > 0:
            score -= min(900, risk * 0.90)
        else:
            score += min(120, -risk * 0.18)
        score += capture_value * 0.05
        if gives_check:
            score += 70
        if rescue:
            score += 150
        if move == main_move:
            score += 80 if self.last_main_depth >= 3 else 25
        if move == mcts_move:
            score += 60
        if verifier._is_bad_opening_capture(board_obj, move, side):
            score -= 260
        score -= verifier._root_repetition_penalty(board_obj, move, side) * 0.80
        return score, safety, risk

    def get_move(self, board_obj, side):
        self.last_main_depth = 0
        self.last_main_partial = 0
        self.last_choice_reason = "main"
        self.last_candidates = 0
        self.last_mcts_move = None

        if self.time_limit < 2.0:
            ai = self._new_minimax(self.time_limit, self.max_depth)
            move = ai.get_move(board_obj, side)
            self.last_main_depth = ai.completed_depth
            self.last_main_partial = ai.partial_depth
            return move

        all_moves = get_all_moves(board_obj, side)
        if not all_moves:
            return None
        if len(all_moves) == 1:
            return all_moves[0]

        probe_budget = min(0.55, max(0.16, self.time_limit * 0.055))
        verify_budget = min(1.85, max(0.42, self.time_limit * 0.17))
        refresh_budget = max(0.12, probe_budget * 0.6)
        mcts_budget = 0.0
        if self.use_mcts and self.time_limit >= 6.0:
            mcts_budget = min(1.65, max(0.55, self.time_limit * 0.14))
        main_budget = max(
            0.55,
            self.time_limit - probe_budget - refresh_budget -
            verify_budget - mcts_budget - 0.18,
        )

        probe_candidates, probe = self._probe_candidates(
            board_obj, side, None, probe_budget)
        legal_set = set(probe._filter_repeating_root_moves(
            board_obj, all_moves, side))

        tactic = probe._find_tactical_finish(board_obj, side, list(legal_set))
        if tactic in legal_set:
            self.last_choice_reason = "tactic"
            return tactic
        defense = probe._find_mate_threat_defense(
            board_obj, side, list(legal_set))
        if defense in legal_set:
            self.last_choice_reason = "mate-defense"
            return defense

        main_ai = self._new_minimax(main_budget, self.max_depth)
        main_move = main_ai.get_move(board_obj, side)
        self.last_main_depth = main_ai.completed_depth
        self.last_main_partial = main_ai.partial_depth
        if main_move not in legal_set:
            main_move = probe_candidates[0] if probe_candidates else all_moves[0]

        mcts_move = self._mcts_proposal(board_obj, side, mcts_budget, legal_set)

        candidates = []
        seen = set()
        for move in (main_move, mcts_move):
            self._add_candidate(move, legal_set, candidates, seen)
        more_candidates, verifier_seed = self._probe_candidates(
            board_obj, side, main_move, refresh_budget)
        for move in probe_candidates + more_candidates:
            self._add_candidate(move, legal_set, candidates, seen)
        self.last_candidates = len(candidates)
        if len(candidates) < 2:
            return main_move

        verifier = self._new_minimax(verify_budget, max_depth=5)
        verifier.tt = verifier_seed.tt
        verifier.history = dict(main_ai.history)
        verifier.deadline = time.time() + max(0.12, verify_budget)
        line_depth = 2 if verify_budget >= 0.75 and self.time_limit >= 4.0 else 1

        scored = {}
        for move in candidates:
            result = self._score_candidate(
                verifier, board_obj, move, side, main_move, mcts_move, line_depth)
            if result is not None:
                scored[move] = result
            if verifier._time_up():
                break

        if main_move not in scored:
            return main_move
        best_move, (best_score, _best_safety, _best_risk) = max(
            scored.items(), key=lambda item: item[1][0])
        main_score, main_safety, main_risk = scored[main_move]
        if best_move == main_move:
            return main_move

        threshold = 155
        if main_risk >= 420 or main_safety < -420:
            threshold = 70
        elif self.last_main_depth < 3:
            threshold = 95
        elif best_move == mcts_move:
            threshold = 180

        if best_score >= main_score + threshold:
            self.last_choice_reason = "verified"
            return best_move
        return main_move


def random_move(board_obj, side):
    moves = get_all_moves(board_obj, side)
    return random.choice(moves) if moves else None


def greedy_move(board_obj, side):
    moves = get_all_moves(board_obj, side)
    if not moves:
        return None
    board = board_obj.board
    best, best_moves = -1, []
    for sx, sy, ex, ey in moves:
        t = board[ey][ex]
        s = piece_values[t[-1]] if t != '.' else 0
        if s > best:
            best, best_moves = s, [(sx, sy, ex, ey)]
        elif s == best:
            best_moves.append((sx, sy, ex, ey))
    return random.choice(best_moves)


# ─────────────────────────────────────────────────────────────────────────────
#  全局实例 & 对外接口（ui.py 不需要改动）
# ─────────────────────────────────────────────────────────────────────────────

#  调整强度：
#    max_depth=3, time_limit=1.0  →  快速，约 0.5-1s
#    max_depth=4, time_limit=2.0  →  均衡，约 1-2s（推荐）
#    max_depth=5, time_limit=3.0  →  强力，约 2-4s

_ai = MinimaxAI(max_depth=8, time_limit=3.0)


def minimax_move(board_obj, side, depth=3):
    """兼容原 ui.py 调用接口，depth 参数保留但不使用"""
    return _ai.get_move(board_obj, side)


def mcts_move(board_obj, side):
    """切换到 MCTS：在 ui.py 中把 minimax_move 替换为 mcts_move"""
    return MCTSAI(time_limit=2.0).get_move(board_obj, side)


def _get_neural_mcts():
    global _neural_mcts
    if _neural_mcts is not None:
        return _neural_mcts
    if not os.path.exists(NEURAL_MODEL_PATH):
        raise FileNotFoundError(
            f"Neural AI model not found: {NEURAL_MODEL_PATH}. "
            "Train a model first or use minimax_move."
        )
    from alpha_mcts import AlphaMCTS
    from network import load_model
    import torch

    device = (
        "cuda"
        if NEURAL_DEVICE == "auto" and torch.cuda.is_available()
        else ("cpu" if NEURAL_DEVICE == "auto" else NEURAL_DEVICE)
    )
    model = load_model(NEURAL_MODEL_PATH, device=device)
    _neural_mcts = AlphaMCTS(
        model, simulations=NEURAL_SIMULATIONS, device=device)
    print(f"[neural AI] loaded {NEURAL_MODEL_PATH}, simulations={NEURAL_SIMULATIONS}")
    return _neural_mcts


def neural_move(board_obj, side, depth=3):
    """Neural-network MCTS move. Does not fall back to minimax."""
    mcts = _get_neural_mcts()
    move, _ = mcts.get_move(board_obj, side, temperature=0, add_noise=False)
    return move
