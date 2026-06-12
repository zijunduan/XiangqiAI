# eval_strength_v2.py
# 更简单的独立棋力评测脚本：
#   1. 兼容旧式 random / greedy / minimax1 / minimax2
#   2. 新增 best_mcts：当前模型 vs chess_model_best.pt 的 MCTS
#   3. 默认不做复杂风险诊断；需要时加 --diagnose
#
# 常用命令：
#   python eval_strength_v2.py --model chess_model.pt --games 10 --sims 64
#   python eval_strength_v2.py --model chess_model.pt --best-model chess_model_best.pt --opponents best_mcts --games 10 --sims 64 --best-sims 64
#   python eval_strength_v2.py --model chess_model.pt --opponents greedy minimax1 best_mcts --games 20 --sims 64
#   python eval_strength_v2.py --model chess_model.pt --model-mode policy --opponents greedy minimax1 --games 20
#
# 说明：
#   - 这个脚本不训练、不保存模型、不改 replay buffer。
#   - model 是“当前待测模型”。
#   - best_model 是“当前最佳模型/旧模型”，用于 best_mcts 对照。
#   - best_mcts 的含义：当前模型 MCTS vs best_model MCTS。
#   - greedy/minimax 默认是旧式材料基准，尽量保持简单直观。

import argparse
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

from board import Board
from network import ChessNet, board_to_tensor, encode_move, _load_state_dict_compatible
from alpha_mcts import AlphaMCTS, get_all_legal, terminal_value
from rules import is_in_check


PIECE_VAL = {
    "j": 10000,
    "c": 900,
    "p": 450,
    "m": 400,
    "x": 200,
    "s": 200,
    "z": 100,
}


# ---------------------------------------------------------------------
# 基础函数
# ---------------------------------------------------------------------

def opponent(side):
    return "r" if side == "b" else "b"


def board_hash(board):
    return tuple(tuple(row) for row in board)


def material_score_black_minus_red(board):
    score = 0
    for row in board:
        for p in row:
            if p == ".":
                continue
            value = PIECE_VAL[p[-1]]
            score += value if p[0] == "b" else -value
    return score


def side_material_score(board, side):
    score = material_score_black_minus_red(board)
    return score if side == "b" else -score


def adjudicate_winner(board, threshold=300):
    score = material_score_black_minus_red(board)
    if score >= threshold:
        return "b"
    if score <= -threshold:
        return "r"
    return None


def terminal_winner(board_obj, side_to_move):
    value = terminal_value(board_obj, side_to_move)
    if value is None:
        return None
    return side_to_move if value > 0 else opponent(side_to_move)


def all_legal(board_obj, side):
    return get_all_legal(board_obj, side)


def captured_value(board_obj, move):
    sx, sy, ex, ey = move
    target = board_obj.board[ey][ex]
    if target == ".":
        return 0
    return PIECE_VAL[target[-1]]


def moving_value(board_obj, move):
    sx, sy, ex, ey = move
    piece = board_obj.board[sy][sx]
    if piece == ".":
        return 0
    return PIECE_VAL[piece[-1]]


def move_gives_check(board_obj, move, side):
    sx, sy, ex, ey = move
    captured = board_obj.move(sx, sy, ex, ey)
    try:
        return is_in_check(board_obj.board, opponent(side))
    finally:
        board_obj.undo_move(sx, sy, ex, ey, captured)


def immediate_recapture_risk(board_obj, move, side):
    """
    诊断用：走完 move 后，对方能否立刻吃掉落点。
    默认不启用，只有 --diagnose 时统计。
    """
    sx, sy, ex, ey = move
    piece = board_obj.board[sy][sx]
    if piece == ".":
        return 0

    value = PIECE_VAL[piece[-1]]
    captured = board_obj.move(sx, sy, ex, ey)
    enemy = opponent(side)

    risk = 0
    try:
        for rsx, rsy, rex, rey in all_legal(board_obj, enemy):
            if rex == ex and rey == ey:
                risk = max(risk, value)
    finally:
        board_obj.undo_move(sx, sy, ex, ey, captured)

    return risk


def is_opening_cannon_rush(board_obj, move, side, ply):
    """
    诊断用：统计开局炮长距离空跑。
    """
    if ply > 4 or side != "r":
        return False

    sx, sy, ex, ey = move
    piece = board_obj.board[sy][sx]
    target = board_obj.board[ey][ex]

    if piece == "." or piece[-1] != "p":
        return False
    if target != ".":
        return False

    dist = abs(ex - sx) + abs(ey - sy)
    return dist >= 5


# ---------------------------------------------------------------------
# 旧式 baseline：random / greedy / minimax
# ---------------------------------------------------------------------

def random_move(board_obj, side):
    legal = all_legal(board_obj, side)
    return random.choice(legal) if legal else None


def greedy_move(board_obj, side):
    """
    旧式贪心：优先吃价值最高的子；没子可吃就随机。
    """
    legal = all_legal(board_obj, side)
    if not legal:
        return None

    best_value = -1
    best_moves = []

    for move in legal:
        value = captured_value(board_obj, move)
        if value > best_value:
            best_value = value
            best_moves = [move]
        elif value == best_value:
            best_moves.append(move)

    return random.choice(best_moves)


def move_order_score(board_obj, move, side):
    cap = captured_value(board_obj, move)
    mov = moving_value(board_obj, move)

    score = cap * 12.0 - mov * 0.03

    if move_gives_check(board_obj, move, side):
        score += 80.0

    return score


def ordered_moves(board_obj, side, width=None):
    legal = all_legal(board_obj, side)
    legal.sort(key=lambda m: move_order_score(board_obj, m, side), reverse=True)
    if width is not None and len(legal) > width:
        return legal[:width]
    return legal


def minimax_eval(board_obj, side_to_move, root_side, depth, width, alpha=-1e18, beta=1e18):
    winner = terminal_winner(board_obj, side_to_move)
    if winner is not None:
        return 30000.0 if winner == root_side else -30000.0

    if depth <= 0:
        return side_material_score(board_obj.board, root_side)

    legal = ordered_moves(board_obj, side_to_move, width)
    if not legal:
        return -30000.0 if side_to_move == root_side else 30000.0

    if side_to_move == root_side:
        best = -1e18
        for move in legal:
            sx, sy, ex, ey = move
            captured = board_obj.move(sx, sy, ex, ey)
            try:
                value = minimax_eval(
                    board_obj,
                    opponent(side_to_move),
                    root_side,
                    depth - 1,
                    width,
                    alpha,
                    beta,
                )
            finally:
                board_obj.undo_move(sx, sy, ex, ey, captured)

            best = max(best, value)
            alpha = max(alpha, best)
            if beta <= alpha:
                break

        return best

    else:
        best = 1e18
        for move in legal:
            sx, sy, ex, ey = move
            captured = board_obj.move(sx, sy, ex, ey)
            try:
                value = minimax_eval(
                    board_obj,
                    opponent(side_to_move),
                    root_side,
                    depth - 1,
                    width,
                    alpha,
                    beta,
                )
            finally:
                board_obj.undo_move(sx, sy, ex, ey, captured)

            best = min(best, value)
            beta = min(beta, best)
            if beta <= alpha:
                break

        return best


def minimax_move(board_obj, side, depth=1, width=32):
    """
    depth=1：只看自己走一步后的材料。
    depth=2：看自己一步 + 对手一步。
    """
    legal = ordered_moves(board_obj, side, width)
    if not legal:
        return None

    best_score = -1e18
    best_moves = []

    for move in legal:
        sx, sy, ex, ey = move
        captured = board_obj.move(sx, sy, ex, ey)
        try:
            winner = terminal_winner(board_obj, opponent(side))
            if winner == side:
                score = 30000.0
            elif winner == opponent(side):
                score = -30000.0
            else:
                score = minimax_eval(
                    board_obj,
                    opponent(side),
                    root_side=side,
                    depth=max(0, depth - 1),
                    width=width,
                )
        finally:
            board_obj.undo_move(sx, sy, ex, ey, captured)

        score += random.random() * 1e-4

        if score > best_score:
            best_score = score
            best_moves = [move]
        elif abs(score - best_score) < 1e-8:
            best_moves.append(move)

    return random.choice(best_moves)


# ---------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------

class RandomAgent:
    name = "random"

    def choose(self, board_obj, side, ply):
        return random_move(board_obj, side)


class GreedyAgent:
    name = "greedy"

    def choose(self, board_obj, side, ply):
        return greedy_move(board_obj, side)


class MinimaxAgent:
    def __init__(self, depth=1, width=32):
        self.depth = depth
        self.width = width
        self.name = f"minimax{depth}"

    def choose(self, board_obj, side, ply):
        return minimax_move(board_obj, side, depth=self.depth, width=self.width)


class NeuralAgent:
    def __init__(self, model, device, mode="mcts", sims=64, name="model"):
        self.model = model
        self.device = device
        self.mode = mode
        self.sims = sims
        self.name = name

        self.mcts = None
        if mode == "mcts":
            self.mcts = AlphaMCTS(model, simulations=sims, device=device)

    def choose(self, board_obj, side, ply):
        if self.mode == "mcts":
            move, _ = self.mcts.get_move(
                board_obj,
                side,
                temperature=0.0,
                add_noise=False,
            )
            return move

        if self.mode == "policy":
            return self.policy_argmax(board_obj, side)

        raise ValueError(f"Unknown neural mode: {self.mode}")

    def policy_argmax(self, board_obj, side):
        legal = all_legal(board_obj, side)
        if not legal:
            return None

        tensor = board_to_tensor(board_obj.board, side).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.inference_mode():
            logits, _ = self.model(tensor)

        idx = torch.tensor(
            [encode_move(*m) for m in legal],
            dtype=torch.long,
            device=self.device,
        )

        legal_logits = logits[0].index_select(0, idx)
        best_i = int(torch.argmax(legal_logits).item())
        return legal[best_i]


def make_baseline_agent(name, minimax_width):
    if name == "random":
        return RandomAgent()
    if name == "greedy":
        return GreedyAgent()
    if name == "minimax1":
        return MinimaxAgent(depth=1, width=minimax_width)
    if name == "minimax2":
        return MinimaxAgent(depth=2, width=minimax_width)
    if name == "minimax3":
        return MinimaxAgent(depth=3, width=minimax_width)
    raise ValueError(f"Unknown baseline opponent: {name}")


# ---------------------------------------------------------------------
# 对局
# ---------------------------------------------------------------------

def play_game(
    red_agent,
    black_agent,
    tested_side,
    max_steps=220,
    rep_after=160,
    rep_thresh=4,
    adjudicate_threshold=300,
    diagnose=False,
    verbose=False,
):
    board_obj = Board()
    side = "r"
    pos_counts = defaultdict(int)

    winner = None
    end_reason = "unknown"

    diag = {
        "opening_cannon_rush": 0,
        "risk400": 0,
        "risk900": 0,
    }

    for ply in range(max_steps):
        tw = terminal_winner(board_obj, side)
        if tw is not None:
            winner = tw
            end_reason = f"terminal, side_to_move={side}"
            break

        h = board_hash(board_obj.board)
        pos_counts[h] += 1
        if ply >= rep_after and pos_counts[h] >= rep_thresh:
            winner = adjudicate_winner(board_obj.board, adjudicate_threshold)
            end_reason = "repetition_adjudicate"
            break

        agent = red_agent if side == "r" else black_agent
        move = agent.choose(board_obj, side, ply)

        if move is None:
            winner = opponent(side)
            end_reason = f"move_none_or_no_legal, side={side}"
            break

        if diagnose and side == tested_side:
            if is_opening_cannon_rush(board_obj, move, side, ply):
                diag["opening_cannon_rush"] += 1

            risk = immediate_recapture_risk(board_obj, move, side)
            if risk >= 400:
                diag["risk400"] += 1
            if risk >= 900:
                diag["risk900"] += 1

        if verbose:
            print(f"ply={ply:3d} side={side} move={move}")

        board_obj.move(*move)
        side = opponent(side)

    else:
        winner = adjudicate_winner(board_obj.board, adjudicate_threshold)
        end_reason = "max_steps_adjudicate"

    if winner is None:
        result = "draw"
    elif winner == tested_side:
        result = "tested_win"
    else:
        result = "tested_loss"

    return {
        "winner": winner,
        "result": result,
        "plies": ply + 1,
        "end_reason": end_reason,
        **diag,
    }


def evaluate_matchup(
    tested_agent,
    opponent_agent,
    games,
    max_steps,
    rep_after,
    rep_thresh,
    adjudicate_threshold,
    diagnose=False,
):
    stats = {
        "tested_win": 0,
        "tested_loss": 0,
        "draw": 0,
        "terminal": 0,
        "repetition": 0,
        "max_steps": 0,
        "other": 0,
        "total_plies": 0,
        "opening_cannon_rush": 0,
        "risk400": 0,
        "risk900": 0,
    }

    t0 = time.time()

    for g in range(games):
        tested_side = "r" if g % 2 == 0 else "b"

        if tested_side == "r":
            red_agent = tested_agent
            black_agent = opponent_agent
        else:
            red_agent = opponent_agent
            black_agent = tested_agent

        result = play_game(
            red_agent,
            black_agent,
            tested_side=tested_side,
            max_steps=max_steps,
            rep_after=rep_after,
            rep_thresh=rep_thresh,
            adjudicate_threshold=adjudicate_threshold,
            diagnose=diagnose,
            verbose=False,
        )

        stats[result["result"]] += 1
        stats["total_plies"] += result["plies"]
        stats["opening_cannon_rush"] += result["opening_cannon_rush"]
        stats["risk400"] += result["risk400"]
        stats["risk900"] += result["risk900"]

        reason = result["end_reason"]
        if "terminal" in reason:
            stats["terminal"] += 1
        elif "repetition" in reason:
            stats["repetition"] += 1
        elif "max_steps" in reason:
            stats["max_steps"] += 1
        else:
            stats["other"] += 1

        print(
            f"  game {g+1:2d}/{games}  tested_side={tested_side}  "
            f"result={result['result']:11s}  winner={result['winner']}  "
            f"plies={result['plies']:3d}  reason={result['end_reason']}"
        )

    elapsed = time.time() - t0
    score = (stats["tested_win"] + 0.5 * stats["draw"]) / max(1, games)
    avg_plies = stats["total_plies"] / max(1, games)

    return stats, score, avg_plies, elapsed


# ---------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------

def load_model(path, channels, n_res, device, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} model file not found: {path}")

    model = ChessNet(channels, n_res).to(device)
    obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "model" in obj:
        state = obj["model"]
    else:
        state = obj

    skipped = _load_state_dict_compatible(model, state)
    if skipped:
        print(f"[load:{label}] skipped {len(skipped)} tensors")
    else:
        print(f"[load:{label}] loaded with no skipped tensors")

    model.eval()
    return model


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="chess_model.pt",
                        help="当前待测模型")
    parser.add_argument("--best-model", type=str, default="chess_model_best.pt",
                        help="当前最佳/旧模型，用于 best_mcts 对照")

    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--n-res", type=int, default=8)

    parser.add_argument("--model-mode", type=str, default="mcts",
                        choices=["mcts", "policy"],
                        help="当前待测模型使用 MCTS 还是纯 policy")
    parser.add_argument("--sims", type=int, default=64,
                        help="当前待测模型 MCTS simulations")
    parser.add_argument("--best-sims", type=int, default=None,
                        help="best_mcts simulations；默认等于 --sims")

    parser.add_argument("--games", type=int, default=20)

    parser.add_argument("--opponents", nargs="+",
                        default=["random", "greedy", "minimax1", "minimax2", "best_mcts"],
                        choices=["random", "greedy", "minimax1", "minimax2", "minimax3", "best_mcts"])

    parser.add_argument("--minimax-width", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--rep-after", type=int, default=160)
    parser.add_argument("--rep-thresh", type=int, default=4)
    parser.add_argument("--adjudicate-threshold", type=int, default=300)

    parser.add_argument("--diagnose", action="store_true",
                        help="开启炮冲/risk400/risk900 诊断统计，稍慢")
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_sims = args.sims if args.best_sims is None else args.best_sims

    print("=" * 80)
    print("Chinese Chess Strength Evaluation v2")
    print("=" * 80)
    print(f"Device        : {device}")
    if device == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"Current model : {args.model}")
    print(f"Best model    : {args.best_model}")
    print(f"Current mode  : {args.model_mode}")
    print(f"Current sims  : {args.sims}")
    print(f"Best sims     : {best_sims}")
    print(f"Games/opponent: {args.games}")
    print(f"Opponents     : {args.opponents}")
    print(f"Diagnose      : {args.diagnose}")
    print("=" * 80)
    print()

    current_model = load_model(args.model, args.channels, args.n_res, device, "current")
    current_agent = NeuralAgent(
        current_model,
        device,
        mode=args.model_mode,
        sims=args.sims,
        name="current",
    )

    best_model = None
    best_agent = None
    if "best_mcts" in args.opponents:
        best_model = load_model(args.best_model, args.channels, args.n_res, device, "best")
        best_agent = NeuralAgent(
            best_model,
            device,
            mode="mcts",
            sims=best_sims,
            name="best_mcts",
        )

    summary = {}

    for opp_name in args.opponents:
        print()
        print("#" * 80)
        print(f"Current model vs {opp_name}")
        print("#" * 80)

        if opp_name == "best_mcts":
            opponent_agent = best_agent
        else:
            opponent_agent = make_baseline_agent(opp_name, args.minimax_width)

        stats, score, avg_plies, elapsed = evaluate_matchup(
            tested_agent=current_agent,
            opponent_agent=opponent_agent,
            games=args.games,
            max_steps=args.max_steps,
            rep_after=args.rep_after,
            rep_thresh=args.rep_thresh,
            adjudicate_threshold=args.adjudicate_threshold,
            diagnose=args.diagnose,
        )

        summary[opp_name] = {
            "stats": stats,
            "score": score,
            "avg_plies": avg_plies,
            "elapsed": elapsed,
        }

        print()
        print("-" * 80)
        print(f"Opponent : {opp_name}")
        print(f"W/L/D    : {stats['tested_win']}/{stats['tested_loss']}/{stats['draw']}")
        print(f"Score    : {score:.1%}")
        print(f"Avg plies: {avg_plies:.1f}")
        print(f"Endings  : terminal={stats['terminal']} repetition={stats['repetition']} "
              f"max_steps={stats['max_steps']} other={stats['other']}")
        if args.diagnose:
            print(f"Diagnose : cannon_rush={stats['opening_cannon_rush']} "
                  f"risk400={stats['risk400']} risk900={stats['risk900']}")
        print(f"Elapsed  : {elapsed:.1f}s")
        print("-" * 80)

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for opp_name, item in summary.items():
        s = item["stats"]
        diag = ""
        if args.diagnose:
            diag = (
                f" | cannon={s['opening_cannon_rush']:3d}"
                f" risk400={s['risk400']:3d}"
                f" risk900={s['risk900']:3d}"
            )

        print(
            f"{opp_name:9s} | "
            f"W={s['tested_win']:2d} "
            f"L={s['tested_loss']:2d} "
            f"D={s['draw']:2d} | "
            f"score={item['score']:6.1%} | "
            f"avg={item['avg_plies']:6.1f} | "
            f"term={s['terminal']:2d} "
            f"rep={s['repetition']:2d}"
            f"{diag}"
        )

    print("=" * 80)
    print()
    print("建议解释：")
    print("  - vs greedy 低，说明材料意识仍弱。")
    print("  - vs minimax1 低于 55%，说明不该扩大 MCTS 自博。")
    print("  - vs best_mcts 低，不必立刻崩溃；它只说明当前模型还没超过旧 best。")
    print("  - 如果 --diagnose 下 cannon/risk900 高，说明仍有炮冲/送大子问题。")
    print()


if __name__ == "__main__":
    main()
