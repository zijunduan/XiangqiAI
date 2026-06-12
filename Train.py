import argparse
import os
import random
import time
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from board import Board
from network import ChessNet, board_to_tensor, encode_move, MOVE_DIM, N_CHANNELS, save_model, _load_state_dict_compatible
from alpha_mcts import AlphaMCTS, get_all_legal, terminal_value
from rules import get_legal_moves, is_in_check

# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "channels"        : 96,
    "n_res"           : 8,
    "simulations"     : 256,
    "simulations_start": 96,
    "sim_ramp_iters"  : 150,
    "games_per_iter"  : 10,
    "max_game_steps"  : 120,
    "mcts_max_steps" : 220,
    "temp_threshold"  : 10,    # 前10步temperature=1探索

    "batch_size"      : 128,
    "train_steps"     : 160,
    "lr"              : 3e-4,
    "weight_decay"    : 1e-4,
    "buffer_size"     : 100000,
    "total_iterations": 500,

    "eval_every"      : 80,
    "eval_start_iter" : 200,
    "eval_games"      : 6,
    "eval_simulations": 64,
    "update_threshold": 0.55,
    "rollback_on_failed_eval": False,

    "save_path"       : "chess_model.pt",
    "best_path"       : "chess_model_best.pt",
    "ckpt_path"       : "chess_ckpt.pt",
    "log_dir"         : "runs/chess",
    "device"          : "cuda" if torch.cuda.is_available() else "cpu",
    "seed"            : 42,
    "data_version"    : 20260615,

    # 重复检测：t >= rep_after 且同一局面出现 rep_thresh 次 → 裁决/平局
    # 训练阶段更早截断循环局，避免长时间生成低信息样本。
    "rep_after"       : 160,
    "rep_thresh"      : 4,
    "adjudicate_threshold": 300,

    # 旧热启动（epsilon-greedy）保留为命令行兼容，默认关闭。
    "init_from_best"  : True,
    "warmup_iters"    : 0,
    "warmup_ratio"    : 0.0,
    "warmup_epsilon"  : 0.20,
    "warmup_max_steps": 180,

    # 快速战术教师：早期用浅战术+反吃检查生成样本，避免弱网络MCTS慢速自我污染。
    "teacher_ratio"   : 0.90,
    "teacher_epsilon" : 0.08,
    "teacher_temp"    : 75.0,
    "teacher_max_steps": 180,
    "teacher_candidate_limit": 32,
    "teacher_decay_iters": 220,
    "teacher_min_ratio": 0.10,

    # 训练早期的弱网络 MCTS 又慢又容易自我污染，先延后、再稀疏混入。
    # 默认每个 MCTS 轮只生成 1 局，避免弱 MCTS 大量重复裁决污染 buffer。
    "mcts_games_per_iter": 1,
    "mcts_start_iter" : 40,
    "mcts_interval"   : 3,
    "mcts_time_budget": 600.0,
    "eval_max_steps"  : 160,
    "tactic_samples_per_iter": 48,

    # 强基样本：真正解决“炮直冲 / 送马炮车 / 打不过一层 minimax”的问题。
    "opening_safety_samples_per_iter": 128,
    "safety_samples_per_iter": 512,
    "expert_depth": 2,
    "expert_width": 24,
    "expert_temperature": 90.0,
    "expert_epsilon": 0.03,

    # 诊断开关：默认关闭，避免每步枚举对方合法走法刷屏。
    "blunder_check": False,
}
# ─────────────────────────────────────────────────────────────────────────────

# 棋子价值表（供 epsilon-greedy 贪心策略使用）
_PIECE_VAL = {'j': 10000, 'c': 900, 'p': 450, 'm': 400, 'x': 200, 's': 200, 'z': 100}


def material_score(board):
    score = 0
    for row in board:
        for p in row:
            if p == '.':
                continue
            value = _PIECE_VAL[p[-1]]
            score += value if p[0] == 'b' else -value
    return score


def adjudicate_winner(board, threshold):
    score = material_score(board)
    if score >= threshold:
        return 'b'
    if score <= -threshold:
        return 'r'
    return None


def current_simulations(iteration, cfg):
    start = cfg.get("simulations_start", cfg["simulations"])
    final = cfg["simulations"]
    ramp = max(1, cfg.get("sim_ramp_iters", 1))
    if iteration >= ramp:
        return final
    ratio = max(0.0, (iteration - 1) / ramp)
    return int(round(start + (final - start) * ratio))


def warmup_games_for_iter(iteration, cfg):
    if iteration > cfg["warmup_iters"]:
        return 0
    progress = (iteration - 1) / max(1, cfg["warmup_iters"])
    ratio = cfg["warmup_ratio"] * max(0.0, 1.0 - progress)
    return round(cfg["games_per_iter"] * ratio)


def teacher_games_for_iter(iteration, cfg):
    ratio0 = cfg.get("teacher_ratio", 0.0)
    decay_iters = cfg.get("teacher_decay_iters", 220)
    min_ratio = cfg.get("teacher_min_ratio", 0.10)

    ratio = ratio0 * max(min_ratio, 1.0 - iteration / decay_iters)

    mcts_games = mcts_games_for_iter(iteration, cfg)
    slots = max(0, cfg["games_per_iter"] - mcts_games)
    return round(slots * ratio)


def mcts_games_for_iter(iteration, cfg):
    if iteration < cfg["mcts_start_iter"]:
        return 0
    if (iteration - cfg["mcts_start_iter"]) % cfg["mcts_interval"] != 0:
        return 0

    override = cfg.get("mcts_games_per_iter", None)
    if override is not None:
        return int(override)

    if iteration < 180:
        return 1
    elif iteration < 300:
        return 3
    else:
        return 5


def opponent(side):
    return 'r' if side == 'b' else 'b'


def board_hash(board):
    return tuple(tuple(r) for r in board)


def _all_legal(board_obj, side):
    """返回 side 的所有合法着法列表"""
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


def side_material_score(board, side):
    score = material_score(board)
    return score if side == 'b' else -score


def bounded_side_value(board, side):
    return float(np.tanh(side_material_score(board, side) / 950.0))


def value_target(board, winner, side):
    if winner == side:
        return 1.0
    if winner == opponent(side):
        return -1.0
    return 0.65 * bounded_side_value(board, side)


def _pawn_progress_bonus(ptype, side, x, y):
    if ptype != 'z':
        return 0
    crossed = (side == 'b' and y >= 5) or (side == 'r' and y <= 4)
    bonus = 22 if crossed else 0
    if crossed and 3 <= x <= 5:
        bonus += 12
    if (side == 'b' and y >= 7) or (side == 'r' and y <= 2):
        bonus += 14
    return bonus


def positional_score(board, side):
    score = 0
    for y in range(10):
        for x in range(9):
            piece = board[y][x]
            if piece == '.':
                continue
            sign = 1 if piece[0] == side else -1
            ptype = piece[-1]
            if ptype in ('c', 'p', 'm') and 2 <= x <= 6 and 2 <= y <= 7:
                score += sign * 18
            score += sign * _pawn_progress_bonus(ptype, piece[0], x, y)
    return score


def teacher_static_score(board, side):
    return side_material_score(board, side) + positional_score(board, side)


def _move_gives_terminal_win(board_obj, enemy):
    terminal = terminal_value(board_obj, enemy)
    return terminal is not None and terminal < 0


def teacher_move_score(board_obj, move, side):
    board = board_obj.board
    sx, sy, ex, ey = move
    piece = board[sy][sx]
    target = board[ey][ex]
    moving_value = _PIECE_VAL[piece[-1]]
    captured_value = _PIECE_VAL[target[-1]] if target != '.' else 0
    enemy = opponent(side)

    captured = board_obj.move(sx, sy, ex, ey)
    try:
        if _move_gives_terminal_win(board_obj, enemy):
            return 20000.0

        score = teacher_static_score(board_obj.board, side)
        if captured != '.':
            score += captured_value * 0.35 - moving_value * 0.04
        if is_in_check(board_obj.board, enemy):
            score += 120

        enemy_moves = _all_legal(board_obj, enemy)
        reply_penalty = 0.0
        worst_after_reply = score
        for reply in enemy_moves:
            rsx, rsy, rex, rey = reply
            reply_target = board_obj.board[rey][rex]
            tactical_reply = reply_target != '.' or (rex == ex and rey == ey)
            if not tactical_reply:
                continue

            reply_capture = board_obj.move(rsx, rsy, rex, rey)
            try:
                if _move_gives_terminal_win(board_obj, side):
                    reply_score = -20000.0
                else:
                    reply_score = teacher_static_score(board_obj.board, side)
                    if is_in_check(board_obj.board, side):
                        reply_score -= 90
                worst_after_reply = min(worst_after_reply, reply_score)
            finally:
                board_obj.undo_move(rsx, rsy, rex, rey, reply_capture)

            if rex == ex and rey == ey:
                loss = max(0, moving_value - captured_value)
                reply_penalty = max(reply_penalty, loss * 0.75)

        return worst_after_reply - reply_penalty
    finally:
        board_obj.undo_move(sx, sy, ex, ey, captured)


def teacher_fast_move_score(board_obj, move, side):
    board = board_obj.board
    sx, sy, ex, ey = move
    piece = board[sy][sx]
    target = board[ey][ex]
    if piece == '.':
        return -100000.0

    moving_value = _PIECE_VAL[piece[-1]]
    captured_value = _PIECE_VAL[target[-1]] if target != '.' else 0
    score = random.random() * 0.01
    if target != '.':
        score += captured_value * 9.0 - moving_value * 0.65
    score += _pawn_progress_bonus(piece[-1], side, ex, ey)
    if piece[-1] in ('c', 'p', 'm') and 2 <= ex <= 6 and 2 <= ey <= 7:
        score += 12
    if piece[-1] == 'j':
        score -= 60
    return score

def immediate_recapture_risk(board_obj, move, side):
    """
    检查 move 走完后，对方是否可以立刻吃掉刚移动到终点的这个子。
    返回可能被直接反吃的己方棋子价值。

    只用于诊断，不改变走法。
    """
    sx, sy, ex, ey = move
    board = board_obj.board
    piece = board[sy][sx]

    if piece == '.':
        return 0

    moving_value = _PIECE_VAL[piece[-1]]
    captured = board_obj.move(sx, sy, ex, ey)
    enemy = opponent(side)

    risk = 0
    try:
        for rsx, rsy, rex, rey in _all_legal(board_obj, enemy):
            if rex == ex and rey == ey:
                risk = max(risk, moving_value)
    finally:
        board_obj.undo_move(sx, sy, ex, ey, captured)

    return risk


def immediate_net_recapture_loss(board_obj, move, side):
    """
    走完 move 后，如果对方下一手能吃掉落点，返回净亏损。
    例如炮吃卒后被吃，净亏约 450-100=350；空走炮被吃，净亏 450。
    """
    sx, sy, ex, ey = move
    board = board_obj.board
    piece = board[sy][sx]
    target = board[ey][ex]
    if piece == '.':
        return 100000.0

    moving_value = _PIECE_VAL[piece[-1]]
    captured_value = _PIECE_VAL[target[-1]] if target != '.' else 0

    captured = board_obj.move(sx, sy, ex, ey)
    enemy = opponent(side)
    worst_loss = 0.0
    try:
        for rsx, rsy, rex, rey in _all_legal(board_obj, enemy):
            if rex == ex and rey == ey:
                worst_loss = max(worst_loss, moving_value - captured_value)
    finally:
        board_obj.undo_move(sx, sy, ex, ey, captured)

    return max(0.0, worst_loss)


def cannon_empty_rush_penalty(board_obj, move):
    """
    专门压制你日志里反复出现的炮空跑到底线：
      (1,7)->(1,0), (7,7)->(7,0)
    它不一定永远是坏棋，但对当前弱网络是强污染源。
    """
    sx, sy, ex, ey = move
    board = board_obj.board
    piece = board[sy][sx]
    target = board[ey][ex]
    if piece == '.' or piece[-1] != 'p':
        return 0.0
    if target != '.':
        return 0.0
    dist = abs(ex - sx) + abs(ey - sy)
    if dist >= 5:
        return 700.0
    return 0.0


def _terminal_winner(board_obj, side_to_move):
    terminal = terminal_value(board_obj, side_to_move)
    if terminal is None:
        return None
    return side_to_move if terminal > 0 else opponent(side_to_move)


def _ordered_legal(board_obj, side, width=None):
    """搜索用走法排序：吃子、将军、中心活子优先。"""
    legal = _all_legal(board_obj, side)
    board = board_obj.board

    def key(move):
        sx, sy, ex, ey = move
        piece = board[sy][sx]
        target = board[ey][ex]
        if piece == '.':
            return -1e9
        moving_value = _PIECE_VAL[piece[-1]]
        captured_value = _PIECE_VAL[target[-1]] if target != '.' else 0
        score = captured_value * 12.0 - moving_value * 0.08
        if piece[-1] in ('c', 'p', 'm') and 2 <= ex <= 6 and 2 <= ey <= 7:
            score += 15.0
        score += _pawn_progress_bonus(piece[-1], side, ex, ey)
        if piece[-1] == 'j':
            score -= 80.0
        return score

    legal.sort(key=key, reverse=True)
    if width is not None and len(legal) > width:
        return legal[:width]
    return legal


def expert_static_eval(board_obj, root_side):
    """不用神经网络，只用材料+位置，从 root_side 视角评估。"""
    return teacher_static_score(board_obj.board, root_side)


def expert_search_eval(board_obj, side_to_move, root_side, depth, width,
                       alpha=-1e18, beta=1e18):
    winner = _terminal_winner(board_obj, side_to_move)
    if winner is not None:
        return 30000.0 if winner == root_side else -30000.0
    if depth <= 0:
        return expert_static_eval(board_obj, root_side)

    legal = _ordered_legal(board_obj, side_to_move, width=width)
    if not legal:
        return -30000.0 if side_to_move == root_side else 30000.0

    maximizing = (side_to_move == root_side)
    if maximizing:
        best = -1e18
        for move in legal:
            sx, sy, ex, ey = move
            captured = board_obj.move(sx, sy, ex, ey)
            try:
                val = expert_search_eval(
                    board_obj, opponent(side_to_move), root_side,
                    depth - 1, width, alpha, beta)
            finally:
                board_obj.undo_move(sx, sy, ex, ey, captured)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best

    best = 1e18
    for move in legal:
        sx, sy, ex, ey = move
        captured = board_obj.move(sx, sy, ex, ey)
        try:
            val = expert_search_eval(
                board_obj, opponent(side_to_move), root_side,
                depth - 1, width, alpha, beta)
        finally:
            board_obj.undo_move(sx, sy, ex, ey, captured)
        best = min(best, val)
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best


def expert_move_score(board_obj, move, side, depth=2, width=24,
                      early_opening=False):
    sx, sy, ex, ey = move
    board = board_obj.board
    piece = board[sy][sx]
    target = board[ey][ex]
    if piece == '.':
        return -100000.0

    moving_value = _PIECE_VAL[piece[-1]]
    captured_value = _PIECE_VAL[target[-1]] if target != '.' else 0

    captured = board_obj.move(sx, sy, ex, ey)
    try:
        winner = _terminal_winner(board_obj, opponent(side))
        if winner == side:
            val = 30000.0
        elif winner == opponent(side):
            val = -30000.0
        else:
            val = expert_search_eval(
                board_obj,
                opponent(side),
                root_side=side,
                depth=max(0, depth - 1),
                width=width,
            )
            if captured != '.':
                val += captured_value * 0.35 - moving_value * 0.04
            if is_in_check(board_obj.board, opponent(side)):
                val += 90.0
    finally:
        board_obj.undo_move(sx, sy, ex, ey, captured)

    net_loss = immediate_net_recapture_loss(board_obj, move, side)
    val -= 1.45 * net_loss
    if early_opening:
        val -= cannon_empty_rush_penalty(board_obj, move)
    return float(val)


def expert_policy(board_obj, legal, side, depth=2, width=24,
                  temperature=90.0, epsilon=0.03, early_opening=False):
    """用 depth-2 材料安全教师生成监督 policy。"""
    if not legal:
        return None, {}

    scores = np.array([
        expert_move_score(board_obj, move, side, depth=depth, width=width,
                          early_opening=early_opening)
        for move in legal
    ], dtype=np.float64)

    best = float(np.max(scores))
    temp = max(1.0, float(temperature))
    logits = np.clip((scores - best) / temp, -16.0, 0.0)
    probs = np.exp(logits)
    s = probs.sum()
    if s <= 0 or not np.isfinite(s):
        probs = np.ones(len(legal), dtype=np.float64) / len(legal)
    else:
        probs /= s

    if epsilon > 0:
        probs = (1.0 - epsilon) * probs + epsilon / len(legal)
        probs /= probs.sum()

    pi = {move: float(prob) for move, prob in zip(legal, probs)}
    move = legal[int(np.random.choice(len(legal), p=probs))]
    return move, pi


def _expert_rollout_position(min_ply, max_ply):
    """用安全教师先走出一个局面，再从该局面生成样本。"""
    board_obj = Board()
    side = 'r'
    n_plies = random.randint(min_ply, max_ply)
    for ply in range(n_plies):
        if _terminal_winner(board_obj, side) is not None:
            break
        legal = _all_legal(board_obj, side)
        if not legal:
            break
        move, _ = expert_policy(
            board_obj, legal, side,
            depth=1, width=18, temperature=120.0, epsilon=0.15,
            early_opening=(ply < 8),
        )
        if move is None:
            break
        board_obj.move(*move)
        side = opponent(side)
    return board_obj, side


def _samples_from_expert_positions(count, min_ply, max_ply, early_opening):
    if count <= 0:
        return []
    samples = []
    for _ in range(count):
        board_obj, side = _expert_rollout_position(min_ply, max_ply)
        if _terminal_winner(board_obj, side) is not None:
            continue
        legal = _all_legal(board_obj, side)
        if not legal:
            continue
        _, pi = expert_policy(
            board_obj, legal, side,
            depth=CFG.get("expert_depth", 2),
            width=CFG.get("expert_width", 24),
            temperature=CFG.get("expert_temperature", 90.0),
            epsilon=CFG.get("expert_epsilon", 0.03),
            early_opening=early_opening,
        )
        pi_np = np.zeros(MOVE_DIM, dtype=np.float32)
        for move, prob in pi.items():
            pi_np[encode_move(*move)] = prob
        board_np = board_to_tensor(board_obj.board, side).numpy()
        z = 0.75 * bounded_side_value(board_obj.board, side)
        samples.append((board_np, pi_np, z))
    return samples


def opening_safety_samples(count):
    """开局反炮冲样本。"""
    return _samples_from_expert_positions(count, min_ply=0, max_ply=8,
                                          early_opening=True)


def safety_teacher_samples(count):
    """中盘/后盘反送子材料安全样本。"""
    return _samples_from_expert_positions(count, min_ply=10, max_ply=80,
                                          early_opening=False)


def teacher_policy(board_obj, legal, side, epsilon, temperature, candidate_limit):
    if len(legal) > candidate_limit:
        ranked = sorted(
            legal,
            key=lambda move: teacher_fast_move_score(board_obj, move, side),
            reverse=True,
        )
        candidates = ranked[:candidate_limit]
        # Keep a little off-policy variety without deep-scoring every move.
        tail = ranked[candidate_limit:]
        if tail:
            candidates.extend(random.sample(tail, min(2, len(tail))))
    else:
        candidates = list(legal)

    scores = np.array(
        [teacher_move_score(board_obj, move, side) for move in candidates],
        dtype=np.float64,
    )
    best = float(np.max(scores))
    temp = max(1.0, float(temperature))
    logits = np.clip((scores - best) / temp, -12.0, 0.0)
    candidate_probs = np.exp(logits)
    candidate_probs /= candidate_probs.sum()

    pi = {m: 0.0 for m in legal}
    for move, prob in zip(candidates, candidate_probs):
        pi[move] = float(prob)

    if epsilon > 0:
        uniform = epsilon / len(legal)
        for move in legal:
            pi[move] = (1.0 - epsilon) * pi[move] + uniform

    total = sum(pi.values()) or 1.0
    pi = {move: prob / total for move, prob in pi.items()}
    moves = list(pi.keys())
    probs = np.array([pi[m] for m in moves], dtype=np.float64)
    move = moves[int(np.random.choice(len(moves), p=probs))]
    return move, pi


TACTIC_POSITIONS = [
    (
        'r',
        [
            (4, 0, 'b_j'), (4, 9, 'r_j'), (4, 6, 'r_z'),
            (1, 5, 'r_c'), (1, 2, 'b_m'),
        ],
    ),
    (
        'b',
        [
            (4, 0, 'b_j'), (4, 9, 'r_j'), (4, 3, 'b_z'),
            (7, 4, 'b_c'), (7, 7, 'r_m'),
        ],
    ),
    (
        'r',
        [
            (4, 0, 'b_j'), (4, 9, 'r_j'), (4, 6, 'r_z'),
            (2, 6, 'r_m'), (2, 1, 'b_c'),
        ],
    ),
    (
        'b',
        [
            (4, 0, 'b_j'), (4, 9, 'r_j'), (4, 3, 'b_z'),
            (6, 3, 'b_m'), (6, 8, 'r_c'),
        ],
    ),
    (
        'r',
        [
            (4, 0, 'b_j'), (4, 9, 'r_j'), (4, 6, 'r_z'),
            (0, 5, 'r_p'), (0, 3, 'b_z'), (0, 1, 'b_c'),
        ],
    ),
    (
        'b',
        [
            (4, 0, 'b_j'), (4, 9, 'r_j'), (4, 3, 'b_z'),
            (8, 4, 'b_p'), (8, 6, 'r_z'), (8, 8, 'r_c'),
        ],
    ),
    (
        'r',
        [
            (4, 0, 'b_j'), (4, 9, 'r_j'), (4, 6, 'r_z'),
            (6, 5, 'r_c'), (6, 1, 'b_m'),
        ],
    ),
    (
        'b',
        [
            (4, 0, 'b_j'), (4, 9, 'r_j'), (4, 3, 'b_z'),
            (2, 4, 'b_c'), (2, 8, 'r_m'),
        ],
    ),
]


def _board_from_pieces(pieces):
    board_obj = Board()
    board_obj.board = [['.' for _ in range(9)] for _ in range(10)]
    for x, y, piece in pieces:
        board_obj.board[y][x] = piece
    return board_obj


def tactic_teacher_samples(count):
    if count <= 0:
        return []

    samples = []
    for _ in range(count):
        side, pieces = random.choice(TACTIC_POSITIONS)
        board_obj = _board_from_pieces(pieces)
        legal = _all_legal(board_obj, side)
        if not legal:
            continue

        _, pi = teacher_policy(
            board_obj, legal, side,
            epsilon=0.0,
            temperature=30.0,
            candidate_limit=64,
        )
        pi_np = np.zeros(MOVE_DIM, dtype=np.float32)
        for move, prob in pi.items():
            pi_np[encode_move(*move)] = prob

        board_np = board_to_tensor(board_obj.board, side).numpy()
        z = value_target(board_obj.board, None, side)
        samples.append((board_np, pi_np, z))

    return samples


def evaluate_tactic_policy(model, device):
    model.eval()
    hits = 0
    total = 0
    for side, pieces in TACTIC_POSITIONS:
        board_obj = _board_from_pieces(pieces)
        legal = _all_legal(board_obj, side)
        if not legal:
            continue

        _, teacher_pi = teacher_policy(
            board_obj, legal, side,
            epsilon=0.0,
            temperature=1.0,
            candidate_limit=64,
        )
        best_teacher = max(teacher_pi.values()) if teacher_pi else 0.0
        acceptable = {
            move for move, prob in teacher_pi.items()
            if prob >= best_teacher * 0.75 and prob > 0.0
        }
        if not acceptable:
            continue

        tensor = board_to_tensor(board_obj.board, side).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits, _ = model(tensor)
        idx = torch.tensor(
            [encode_move(*move) for move in legal],
            dtype=torch.long,
            device=device,
        )
        probs = F.softmax(logits[0].index_select(0, idx), dim=0)
        best_i = int(torch.argmax(probs).item())
        if legal[best_i] in acceptable:
            hits += 1
        total += 1

    return hits / max(1, total), hits, total


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, scaler,
                    iteration, global_step, replay_buffer, cfg):
    buf_snap = list(replay_buffer)[-50000:]
    torch.save({
        "iteration"    : iteration,
        "global_step"  : global_step,
        "model"        : model.state_dict(),
        "optimizer"    : optimizer.state_dict(),
        "scheduler"    : scheduler.state_dict(),
        "scaler"       : scaler.state_dict(),
        "replay_buffer": buf_snap,
        "data_version" : cfg.get("data_version", 1),
    }, cfg["ckpt_path"])


def load_checkpoint(model, optimizer, scheduler, scaler, cfg, device):
    if not os.path.exists(cfg["ckpt_path"]):
        print("[冷启动] 无存档，从第 1 轮开始")
        return 1, 0, []
    try:
        ckpt = torch.load(cfg["ckpt_path"], map_location=device,weights_only=False)
        if ckpt.get("data_version") != cfg.get("data_version", 1):
            print("[resume] MCTS/data version changed; replay buffer reset")
            if cfg.get("init_from_best") and os.path.exists(cfg["best_path"]):
                skipped = _load_state_dict_compatible(
                    model, torch.load(cfg["best_path"], map_location=device))
                if skipped:
                    print(f"[resume] compatible best load skipped {len(skipped)} tensors")
            return 1, 0, []
        skipped = _load_state_dict_compatible(model, ckpt["model"])
        if skipped:
            print(f"[resume] compatible model load skipped {len(skipped)} tensors")
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as opt_e:
            print(f"[resume] optimizer state reset after model change: {opt_e}")
        buf = [
            sample for sample in ckpt.get("replay_buffer", [])
            if len(sample) >= 3 and getattr(sample[0], "shape", None) == (N_CHANNELS, 10, 9)
        ]
        print(f"[续训] 第 {ckpt['iteration']+1} 轮  "
              f"global_step={ckpt['global_step']}  buffer={len(buf)}")
        return ckpt["iteration"] + 1, ckpt["global_step"], buf
    except Exception as e:
        print(f"[警告] 存档读取失败，冷启动: {e}")
        return 1, 0, []


# ── 热启动：epsilon-greedy 棋谱生成 ─────────────────────────────────────────

def _eg_move(board_obj, side, epsilon):
    """
    epsilon-greedy 走法：
      以 epsilon 概率随机走，以 1-epsilon 概率贪心选最大吃子。
    相比纯随机：游戏更快收敛出胜负。
    相比纯确定性minimax：不陷入固定循环，产生多样棋谱。
    """
    moves = _all_legal(board_obj, side)
    if not moves:
        return None
    if random.random() < epsilon:
        return random.choice(moves)
    # 贪心：按吃子价值排序，随机打平（避免完全确定性）
    board = board_obj.board
    best_val = -1
    best = []
    for m in moves:
        t = board[m[3]][m[2]]
        v = _PIECE_VAL[t[-1]] if t != '.' else 0
        if v > best_val:
            best_val = v
            best = [m]
        elif v == best_val:
            best.append(m)
    return random.choice(best)


def warmup_game(max_steps, rep_after, rep_thresh, epsilon):
    """
    用 epsilon-greedy 生成一局热启动棋谱。
    返回 [(board_np, pi_np, z), ...]，pi 用合法着法均匀分布。
    """
    board_obj  = Board()
    side       = 'r'
    history    = []
    winner     = None
    pos_counts = defaultdict(int)

    for step in range(max_steps):
        terminal = terminal_value(board_obj, side)
        if terminal is not None:
            winner = side if terminal > 0 else ('b' if side == 'r' else 'r')
            break
        h = board_hash(board_obj.board)
        pos_counts[h] += 1
        if step >= rep_after and pos_counts[h] >= rep_thresh:
            winner = adjudicate_winner(board_obj.board, CFG["adjudicate_threshold"])
            break

        legal = _all_legal(board_obj, side)
        if not legal:
            winner = 'b' if side == 'r' else 'r'
            break

        move = _eg_move(board_obj, side, epsilon)
        board_np = board_to_tensor(board_obj.board, side).numpy()
        history.append((board_np, legal, move, side))
        board_obj.move(*move)
        side = 'r' if side == 'b' else 'b'
    else:
        winner = adjudicate_winner(board_obj.board, CFG["adjudicate_threshold"])

    samples = []
    for board_np, legal, move, s in history:
        pi_np = np.zeros(MOVE_DIM, dtype=np.float32)
        smoothing = 0.10
        w = smoothing / len(legal)
        for sx, sy, ex, ey in legal:
            pi_np[encode_move(sx, sy, ex, ey)] = w
        pi_np[encode_move(*move)] += 1.0 - smoothing
        z = value_target(board_obj.board, winner, s)
        samples.append((board_np, pi_np, z))

    return samples, winner


def teacher_game(max_steps, rep_after, rep_thresh, epsilon, temperature,
                 candidate_limit):
    """
    用快速战术教师生成一局棋谱。
    pi 是教师软分布，z 在无明确胜负时使用有界材料评估，减少纯平局退化。
    """
    board_obj  = Board()
    side       = 'r'
    history    = []
    winner     = None
    pos_counts = defaultdict(int)

    for step in range(max_steps):
        terminal = terminal_value(board_obj, side)
        if terminal is not None:
            winner = side if terminal > 0 else opponent(side)
            break
        h = board_hash(board_obj.board)
        pos_counts[h] += 1
        if step >= rep_after and pos_counts[h] >= rep_thresh:
            winner = adjudicate_winner(board_obj.board, CFG["adjudicate_threshold"])
            break

        legal = _all_legal(board_obj, side)
        if not legal:
            winner = opponent(side)
            break

        move, pi = teacher_policy(
            board_obj, legal, side, epsilon, temperature, candidate_limit)
        board_np = board_to_tensor(board_obj.board, side).numpy()
        history.append((board_np, pi, side))
        board_obj.move(*move)
        side = opponent(side)
    else:
        winner = adjudicate_winner(board_obj.board, CFG["adjudicate_threshold"])

    samples = []
    for board_np, pi, s in history:
        pi_np = np.zeros(MOVE_DIM, dtype=np.float32)
        for m, p in pi.items():
            pi_np[encode_move(*m)] = p
        samples.append((board_np, pi_np, value_target(board_obj.board, winner, s)))

    return samples, winner


# ── AlphaMCTS 自对弈 ──────────────────────────────────────────────────────────

def mcts_game(mcts, max_steps, temp_threshold, rep_after, rep_thresh,
              time_budget=None):
    board_obj  = Board()
    side       = 'r'
    history    = []
    winner     = None
    end_reason = "unknown"
    pos_counts = defaultdict(int)
    start_time = time.time()

    for step in range(max_steps):
        # 1. 时间预算
        if time_budget is not None and time.time() - start_time >= time_budget:
            winner = adjudicate_winner(board_obj.board, CFG["adjudicate_threshold"])
            end_reason = "time_budget"
            break

        # 2. 终局检测
        terminal = terminal_value(board_obj, side)
        if terminal is not None:
            winner = side if terminal > 0 else opponent(side)
            end_reason = f"terminal_value={terminal}, side={side}"
            break

        # 3. 重复局面检测
        h = board_hash(board_obj.board)
        pos_counts[h] += 1
        if step >= rep_after and pos_counts[h] >= rep_thresh:
            winner = adjudicate_winner(board_obj.board, CFG["adjudicate_threshold"])
            end_reason = "repetition_adjudicate"
            break

        # 4. 合法走法检测
        legal = get_all_legal(board_obj, side)
        if not legal:
            winner = opponent(side)
            end_reason = f"no_legal_moves, side={side}"
            break

        # 5. MCTS 选步
        temperature = 1.0 if step < temp_threshold else 0.0
        move, pi = mcts.get_move(
            board_obj,
            side,
            temperature=temperature,
            add_noise=(step < temp_threshold),
        )

        if move is None:
            winner = opponent(side)
            end_reason = f"move_none, side={side}"
            break

        # 6. 送子诊断：只打印，不干预走法。默认关闭，避免额外枚举合法走法。
        if CFG.get("blunder_check", False):
            risk = immediate_recapture_risk(board_obj, move, side)
            if risk >= 400:
                print(
                    f"  [BLUNDER?] MC step={step} side={side} move={move} risk={risk}",
                    flush=True,
                )

        # 7. 保存训练样本
        board_np = board_to_tensor(board_obj.board, side).numpy()
        pi_np = np.zeros(MOVE_DIM, dtype=np.float32)
        for m, p in pi.items():
            pi_np[encode_move(*m)] = p

        history.append((board_np, pi_np, side))

        # 8. 实际走子
        board_obj.move(*move)
        side = opponent(side)

    else:
        winner = adjudicate_winner(board_obj.board, CFG["adjudicate_threshold"])
        end_reason = "max_steps_adjudicate"

    samples = []
    for board_np, pi_np, s in history:
        z = value_target(board_obj.board, winner, s)
        samples.append((board_np, pi_np, z))

    return samples, winner, end_reason


# ── 训练步 ────────────────────────────────────────────────────────────────────

def train_step(model, optimizer, scaler, replay_buffer, batch_size, device):
    if len(replay_buffer) < batch_size:
        return None, None

    batch = random.sample(list(replay_buffer), batch_size)
    board_nps, pi_nps, z_vals = zip(*batch)

    boards = torch.from_numpy(np.stack(board_nps)).to(device)
    pis    = torch.from_numpy(np.stack(pi_nps)).to(device)
    zs     = torch.tensor(z_vals, dtype=torch.float32, device=device)

    model.train()
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        policy_logits, values = model(boards)
        loss_p = -(pis * F.log_softmax(policy_logits, dim=1)).sum(1).mean()
        loss_v = F.mse_loss(values.squeeze(1), zs)
        loss   = loss_p + loss_v

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss_p.item(), loss_v.item()


# ── 模型评估 ─────────────────────────────────────────────────────────────────

def evaluate_models(new_model, old_model, n_games, simulations, device,
                    rep_after, rep_thresh, max_steps=96):
    new_mcts = AlphaMCTS(new_model, simulations=simulations, device=device)
    old_mcts = AlphaMCTS(old_model, simulations=simulations, device=device)
    new_wins = 0
    draws = 0
    new_score = 0.0

    for game_i in range(n_games):
        new_is_red = (game_i % 2 == 0)
        red_mcts   = new_mcts if new_is_red else old_mcts
        blk_mcts   = old_mcts if new_is_red else new_mcts

        board_obj  = Board()
        side       = 'r'
        winner     = None
        pos_counts = defaultdict(int)

        for step_e in range(max_steps):
            terminal = terminal_value(board_obj, side)
            if terminal is not None:
                winner = side if terminal > 0 else ('b' if side == 'r' else 'r')
                break
            h = board_hash(board_obj.board)
            pos_counts[h] += 1
            if step_e >= rep_after and pos_counts[h] >= rep_thresh:
                winner = adjudicate_winner(board_obj.board, CFG["adjudicate_threshold"])
                break
            cur = red_mcts if side == 'r' else blk_mcts
            legal = get_all_legal(board_obj, side)
            if not legal:
                winner = 'b' if side == 'r' else 'r'
                break
            move, _ = cur.get_move(board_obj, side, temperature=0)
            if move is None:
                winner = 'b' if side == 'r' else 'r'
                break
            board_obj.move(*move)
            side = 'r' if side == 'b' else 'b'
        else:
            winner = adjudicate_winner(board_obj.board, CFG["adjudicate_threshold"])

        if winner is None:
            draws += 1
            new_score += 0.5
        elif (winner == 'r' and new_is_red) or (winner == 'b' and not new_is_red):
            new_wins += 1
            new_score += 1.0

    return new_score / max(1, n_games), new_wins, draws


# ── 主循环 ────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    random.seed(CFG["seed"])

    device = CFG["device"]
    print(f"[训练] 设备: {device}")
    if device == "cuda":
        print(f"[训练] GPU : {torch.cuda.get_device_name(0)}")
        print(f"[训练] 显存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    writer    = SummaryWriter(log_dir=CFG["log_dir"])
    model     = ChessNet(CFG["channels"], CFG["n_res"]).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=CFG["total_iterations"], eta_min=1e-5)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    replay_buffer = deque(maxlen=CFG["buffer_size"])

    if args.resume:
        start_iter, global_step, buf = load_checkpoint(
            model, optimizer, scheduler, scaler, CFG, device)
        replay_buffer.extend(buf)
    else:
        start_iter, global_step = 1, 0
        if CFG["init_from_best"] and os.path.exists(CFG["best_path"]):
            skipped = _load_state_dict_compatible(
                model, torch.load(CFG["best_path"], map_location=device))
            if skipped:
                print(f"[init] compatible best load skipped {len(skipped)} tensors")
            print(f"[init] 从 {CFG['best_path']} 初始化，replay buffer 清空")

    mcts = AlphaMCTS(model, simulations=CFG["simulations"], device=device)

    for iteration in range(start_iter, CFG["total_iterations"] + 1):
        sim_now = current_simulations(iteration, CFG)
        mcts.simulations = sim_now
        print(f"\n{'='*50}  第 {iteration}/{CFG['total_iterations']} 轮  {'='*50}")
        n_mcts = mcts_games_for_iter(iteration, CFG)
        n_teacher = teacher_games_for_iter(iteration, CFG)
        n_warmup = min(
            warmup_games_for_iter(iteration, CFG),
            max(0, CFG["games_per_iter"] - n_teacher - n_mcts),
        )
        print(f"  MCTS simulations={sim_now}  games={CFG['games_per_iter']}  "
              f"teacher_games={n_teacher}  mcts_games={n_mcts}  "
              f"train_steps={CFG['train_steps']}  "
              f"tc_steps={CFG['teacher_max_steps']}  "
              f"mc_steps={CFG['mcts_max_steps']}  "
              f"mc_budget={CFG.get('mcts_time_budget')}s")

        # ── 1. 棋谱生成 ───────────────────────────────────────────────────────
        t0         = time.time()
        win_counts = {'r': 0, 'b': 0, None: 0}
        tc_count   = 0
        wu_count   = 0
        tc_steps_list = []
        mc_steps_list = []

        for g in range(CFG["games_per_iter"]):
            use_teacher = (g < n_teacher)
            use_warmup = (n_teacher <= g < n_teacher + n_warmup)
            use_mcts = (g >= CFG["games_per_iter"] - n_mcts)

            if use_teacher:
                samples, winner = teacher_game(
                    max_steps = CFG["teacher_max_steps"],
                    rep_after = CFG["rep_after"],
                    rep_thresh= CFG["rep_thresh"],
                    epsilon   = CFG["teacher_epsilon"],
                    temperature= CFG["teacher_temp"],
                    candidate_limit=CFG["teacher_candidate_limit"],
                )
                tc_count += 1
                tag = "[TC]"
            elif use_warmup:
                samples, winner = warmup_game(
                    max_steps = CFG["warmup_max_steps"],
                    rep_after = CFG["rep_after"],
                    rep_thresh= CFG["rep_thresh"],
                    epsilon   = CFG["warmup_epsilon"],
                )
                wu_count += 1
                tag = "[WU]"
            elif use_mcts:
                samples, winner, end_reason = mcts_game(
                    mcts,
                    max_steps=CFG["mcts_max_steps"],
                    temp_threshold=CFG["temp_threshold"],
                    rep_after=CFG["rep_after"],
                    rep_thresh=CFG["rep_thresh"],
                    time_budget=CFG.get("mcts_time_budget"),
                )
                tag = "[MC]"
            else:
                samples, winner = teacher_game(
                    max_steps = CFG["teacher_max_steps"],
                    rep_after = CFG["rep_after"],
                    rep_thresh= CFG["rep_thresh"],
                    epsilon   = CFG["teacher_epsilon"],
                    temperature= CFG["teacher_temp"],
                    candidate_limit=CFG["teacher_candidate_limit"],
                )
                tc_count += 1
                tag = "[TC]"

            replay_buffer.extend(samples)
            win_counts[winner] += 1

            if tag == "[TC]":
                tc_steps_list.append(len(samples))
            elif tag == "[MC]":
                mc_steps_list.append(len(samples))

            nz = sum(1 for s in samples if s[2] != 0.0)

            extra = ""
            if tag == "[MC]":
                extra = f"  reason={end_reason}"

            print(f"  [{g+1:2d}/{CFG['games_per_iter']}]{tag} "
                  f"步={len(samples):3d}  "
                  f"胜={'平' if winner is None else winner}  "
                  f"非零z={nz}/{len(samples)}  "
                  f"buf={len(replay_buffer)}{extra}", flush=True)
        tactic_samples = tactic_teacher_samples(CFG.get("tactic_samples_per_iter", 0))
        if tactic_samples:
            replay_buffer.extend(tactic_samples)
            writer.add_scalar("selfplay/tactic_samples", len(tactic_samples), iteration)
            print(f"  [TACTIC] samples={len(tactic_samples)}  buf={len(replay_buffer)}",
                  flush=True)

        opening_samples = opening_safety_samples(
            CFG.get("opening_safety_samples_per_iter", 0))
        if opening_samples:
            replay_buffer.extend(opening_samples)
            writer.add_scalar("selfplay/opening_safety_samples", len(opening_samples), iteration)
            print(f"  [OPENING_SAFE] samples={len(opening_samples)}  buf={len(replay_buffer)}",
                  flush=True)

        safety_samples = safety_teacher_samples(
            CFG.get("safety_samples_per_iter", 0))
        if safety_samples:
            replay_buffer.extend(safety_samples)
            writer.add_scalar("selfplay/safety_samples", len(safety_samples), iteration)
            print(f"  [SAFETY] samples={len(safety_samples)}  buf={len(replay_buffer)}",
                  flush=True)

        if device == "cuda":
            torch.cuda.empty_cache()

        buf_nz = sum(1 for _, _, z in replay_buffer if z != 0.0) / max(len(replay_buffer), 1)
        gpi    = CFG["games_per_iter"]

        writer.add_scalar("selfplay/red_win_rate",   win_counts['r'] / gpi, iteration)
        writer.add_scalar("selfplay/black_win_rate", win_counts['b'] / gpi, iteration)
        writer.add_scalar("selfplay/draw_rate",      win_counts[None] / gpi, iteration)
        writer.add_scalar("selfplay/buf_nonzero_z",  buf_nz,                 iteration)
        all_steps = tc_steps_list + mc_steps_list
        tc_avg = float(np.mean(tc_steps_list)) if tc_steps_list else 0.0
        mc_avg = float(np.mean(mc_steps_list)) if mc_steps_list else 0.0
        short_count = sum(1 for s in all_steps if s < 60)
        short_rate = short_count / max(1, len(all_steps))

        writer.add_scalar("selfplay/tc_avg_steps", tc_avg, iteration)
        writer.add_scalar("selfplay/mc_avg_steps", mc_avg, iteration)
        writer.add_scalar("selfplay/short_game_rate", short_rate, iteration)

        print(f"  步数统计 | TC_avg={tc_avg:.1f} "
              f"MC_avg={mc_avg:.1f} 短局<60={short_count}/{len(all_steps)}")
        print(f"  耗时={time.time()-t0:.1f}s | "
              f"红胜={win_counts['r']} 黑胜={win_counts['b']} 平={win_counts[None]} | "
              f"TC局={tc_count} WU局={wu_count} | buf非零z={buf_nz:.1%}")

        if buf_nz < 0.05 and iteration > 5:
            print("  [WARN] buf非零z<5%，value head 面临退化风险！"
                  "可用 --teacher-ratio 提高教师棋谱比例。")

        # ── 2. 训练 ───────────────────────────────────────────────────────────
        if len(replay_buffer) < CFG["batch_size"]:
            print("  [跳过] 样本不足")
            save_checkpoint(model, optimizer, scheduler, scaler,
                            iteration, global_step, replay_buffer, CFG)
            continue

        t1 = time.time()
        p_losses, v_losses = [], []
        for _ in range(CFG["train_steps"]):
            pl, vl = train_step(model, optimizer, scaler,
                                replay_buffer, CFG["batch_size"], device)
            if pl is not None:
                p_losses.append(pl)
                v_losses.append(vl)
                global_step += 1
                if global_step % 100 == 0:
                    writer.add_scalar("loss/policy", pl, global_step)
                    writer.add_scalar("loss/value",  vl, global_step)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        if p_losses:
            mp, mv_ = np.mean(p_losses), np.mean(v_losses)
            writer.add_scalar("loss/policy_epoch", mp,  iteration)
            writer.add_scalar("loss/value_epoch",  mv_, iteration)
            writer.add_scalar("train/lr",          lr,  iteration)
            print(f"  训练{CFG['train_steps']}步 | "
                  f"policy={mp:.4f}  value={mv_:.4f}  "
                  f"lr={lr:.2e}  耗时={time.time()-t1:.1f}s")
            if mv_ < 0.002 and buf_nz < 0.1:
                print("  [WARN] value_loss接近0 且非零z<10%：value head 退化！")

        # ── 3. 评估与模型更新 ─────────────────────────────────────────────────
        do_eval = (
            iteration >= CFG.get("eval_start_iter", CFG["eval_every"]) and
            iteration % CFG["eval_every"] == 0
        )

        if do_eval and os.path.exists(CFG["best_path"]):
            print(f"  评估（{CFG['eval_games']}局）...")
            old_model = ChessNet(CFG["channels"], CFG["n_res"]).to(device)
            skipped = _load_state_dict_compatible(
                old_model, torch.load(CFG["best_path"], map_location=device))
            if skipped:
                print(f"[eval] compatible best load skipped {len(skipped)} tensors")
            old_model.eval()

            tactic_acc, tactic_hits, tactic_total = evaluate_tactic_policy(
                model, device)
            old_tactic_acc, old_tactic_hits, old_tactic_total = evaluate_tactic_policy(
                old_model, device)
            writer.add_scalar("eval/tactic_policy_acc", tactic_acc, iteration)

            wr, eval_wins, eval_draws = evaluate_models(
                model, old_model,
                n_games    = CFG["eval_games"],
                simulations= CFG["eval_simulations"],
                device     = device,
                rep_after  = CFG["rep_after"],
                rep_thresh = CFG["rep_thresh"],
                max_steps  = CFG["eval_max_steps"],
            )
            del old_model
            if device == "cuda":
                torch.cuda.empty_cache()

            writer.add_scalar("eval/win_rate_vs_best", wr, iteration)
            print(f"  战术policy={tactic_hits}/{tactic_total} "
                  f"(best={old_tactic_hits}/{old_tactic_total})")
            print(f"  新网络评估分={wr:.1%}  "
                  f"wins={eval_wins}/{CFG['eval_games']} "
                  f"draws={eval_draws}/{CFG['eval_games']} "
                  f"(阈值={CFG['update_threshold']:.0%})")

            tactic_not_worse = tactic_acc + 0.05 >= old_tactic_acc
            if wr >= CFG["update_threshold"] and tactic_not_worse:
                save_model(model, CFG["save_path"])
                save_model(model, CFG["best_path"])
                save_model(model, f"chess_model_iter{iteration}.pt")
                print("  [OK] 更新最优网络")
            else:
                if wr >= CFG["update_threshold"] and not tactic_not_worse:
                    print("  [KEEP] 对战分够但战术退步，暂不更新 best")
                elif CFG.get("rollback_on_failed_eval", False):
                    print("  [ROLLBACK] 回退至最优网络")
                    skipped = _load_state_dict_compatible(
                        model, torch.load(CFG["best_path"], map_location=device))
                    if skipped:
                        print(f"[eval] compatible rollback skipped {len(skipped)} tensors")
                    mcts = AlphaMCTS(model,
                                     simulations=sim_now, device=device)
                else:
                    print("  [KEEP] 未更新 best，继续训练候选网络")
                save_model(model, CFG["save_path"])
        else:
            save_model(model, CFG["save_path"])
            if not os.path.exists(CFG["best_path"]):
                save_model(model, CFG["best_path"])

        save_checkpoint(model, optimizer, scheduler, scaler,
                        iteration, global_step, replay_buffer, CFG)
        writer.flush()

    writer.close()
    print("\n训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--games",       type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--cold-start",  action="store_true")
    parser.add_argument("--warmup",      type=int, default=None,
                        help="覆盖热启动轮数 warmup_iters")
    parser.add_argument("--teacher-ratio", type=float, default=None,
                        help="覆盖战术教师初始比例 teacher_ratio")
    parser.add_argument("--teacher-candidates", type=int, default=None,
                        help="覆盖每步深评分候选数 teacher_candidate_limit")
    parser.add_argument("--teacher-max-steps", type=int, default=None,
                        help="override teacher game max plies")
    parser.add_argument("--mcts-start", type=int, default=None,
                        help="覆盖训练 MCTS 自搏开始轮数 mcts_start_iter")
    parser.add_argument("--mcts-interval", type=int, default=None,
                        help="覆盖训练 MCTS 自搏间隔 mcts_interval")
    parser.add_argument("--mcts-games", type=int, default=None,
                        help="override MCTS self-play games per MCTS round")
    parser.add_argument("--mcts-max-steps", type=int, default=None,
                        help="override MCTS self-play max plies")
    parser.add_argument("--max-game-steps", type=int, default=None,
                        help="compat alias for --mcts-max-steps")
    parser.add_argument("--mcts-time-budget", type=float, default=None,
                        help="override MCTS self-play seconds per game")
    parser.add_argument("--temp-threshold", type=int, default=None,
                        help="override opening plies with MCTS temperature/noise")
    parser.add_argument("--eval-games", type=int, default=None,
                        help="override evaluation game count")
    parser.add_argument("--eval-simulations", type=int, default=None,
                        help="override evaluation simulations per move")
    parser.add_argument("--eval-max-steps", type=int, default=None,
                        help="override max plies per evaluation game")
    parser.add_argument("--eval-start", type=int, default=None,
                        help="override first iteration that runs evaluation")
    parser.add_argument("--iterations", type=int, default=None,
                        help="override total training iterations")
    parser.add_argument("--tactic-samples", type=int, default=None,
                        help="override fixed tactical teacher samples per iteration")
    parser.add_argument("--opening-safety-samples", type=int, default=None,
                        help="override opening anti-blunder expert samples per iteration")
    parser.add_argument("--safety-samples", type=int, default=None,
                        help="override material safety expert samples per iteration")
    parser.add_argument("--expert-depth", type=int, default=None,
                        help="override expert search depth for safety samples")
    parser.add_argument("--expert-width", type=int, default=None,
                        help="override expert search width for safety samples")
    parser.add_argument("--blunder-check", action="store_true",
                        help="enable expensive BLUNDER diagnostic prints")
    parser.add_argument("--rollback-on-failed-eval", action="store_true",
                        help="restore old behavior: rollback to best after failed eval")
    args = parser.parse_args()
    if args.simulations:
        CFG["simulations"] = args.simulations
        CFG["simulations_start"] = min(CFG["simulations_start"], args.simulations)
    if args.games:
        CFG["games_per_iter"] = args.games
    if args.train_steps:
        CFG["train_steps"] = args.train_steps
    if args.cold_start:
        CFG["init_from_best"] = False
    if args.warmup is not None:
        CFG["warmup_iters"] = args.warmup
    if args.teacher_ratio is not None:
        CFG["teacher_ratio"] = max(0.0, min(1.0, args.teacher_ratio))
    if args.teacher_candidates is not None:
        CFG["teacher_candidate_limit"] = max(4, args.teacher_candidates)
    if args.teacher_max_steps is not None:
        CFG["teacher_max_steps"] = max(20, args.teacher_max_steps)
    if args.mcts_start is not None:
        CFG["mcts_start_iter"] = max(1, args.mcts_start)
    if args.mcts_interval is not None:
        CFG["mcts_interval"] = max(1, args.mcts_interval)
    if args.mcts_games is not None:
        CFG["mcts_games_per_iter"] = max(0, args.mcts_games)
    mcts_max_steps = args.mcts_max_steps
    if args.max_game_steps is not None:
        mcts_max_steps = args.max_game_steps
    if mcts_max_steps is not None:
        CFG["mcts_max_steps"] = max(20, mcts_max_steps)
        CFG["max_game_steps"] = CFG["mcts_max_steps"]
    if args.mcts_time_budget is not None:
        CFG["mcts_time_budget"] = max(1.0, args.mcts_time_budget)
    if args.temp_threshold is not None:
        CFG["temp_threshold"] = max(0, args.temp_threshold)
    if args.eval_games is not None:
        CFG["eval_games"] = max(2, args.eval_games)
    if args.eval_simulations is not None:
        CFG["eval_simulations"] = max(1, args.eval_simulations)
    if args.eval_max_steps is not None:
        CFG["eval_max_steps"] = max(20, args.eval_max_steps)
    if args.eval_start is not None:
        CFG["eval_start_iter"] = max(1, args.eval_start)
    if args.iterations is not None:
        CFG["total_iterations"] = max(1, args.iterations)
    if args.tactic_samples is not None:
        CFG["tactic_samples_per_iter"] = max(0, args.tactic_samples)
    if args.opening_safety_samples is not None:
        CFG["opening_safety_samples_per_iter"] = max(0, args.opening_safety_samples)
    if args.safety_samples is not None:
        CFG["safety_samples_per_iter"] = max(0, args.safety_samples)
    if args.expert_depth is not None:
        CFG["expert_depth"] = max(1, args.expert_depth)
    if args.expert_width is not None:
        CFG["expert_width"] = max(4, args.expert_width)
    if args.blunder_check:
        CFG["blunder_check"] = True
    if args.rollback_on_failed_eval:
        CFG["rollback_on_failed_eval"] = True
    main(args)
