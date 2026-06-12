"""
alpha_mcts.py — PUCT-based MCTS using the neural network

Key fixes vs original:
  1. Board state is fully restored after each simulation (no cross-batch corruption)
  2. Batch simulation replaced with single-step simulation (simpler, correct)
  3. Value is flipped correctly on backup relative to the *root* side
  4. Terminal detection returns proper values (not always 1.0)
"""

import math
import copy
import numpy as np
import torch
import torch.nn.functional as F

from rules import get_legal_moves, is_in_check, find_jiang, is_attacked
from network import board_to_tensor, encode_move, MOVE_DIM

PIECE_VAL = {'j': 10000, 'c': 900, 'p': 450, 'm': 400, 'x': 200, 's': 200, 'z': 100}
MATE_SCORE = 20000.0
NETWORK_POLICY_WEIGHT = 0.30
NETWORK_VALUE_WEIGHT = 0.30


def opponent(side):
    return 'r' if side == 'b' else 'b'


def get_all_legal(board_obj, side):
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


def _pawn_progress_bonus(ptype, side, x, y):
    if ptype != 'z':
        return 0
    crossed = (side == 'b' and y >= 5) or (side == 'r' and y <= 4)
    bonus = 18 if crossed else 0
    if crossed and 3 <= x <= 5:
        bonus += 10
    if (side == 'b' and y >= 7) or (side == 'r' and y <= 2):
        bonus += 12
    return bonus


def raw_side_score(board, side):
    score = 0.0
    for y in range(10):
        for x in range(9):
            piece = board[y][x]
            if piece == '.':
                continue
            sign = 1.0 if piece[0] == side else -1.0
            ptype = piece[-1]
            score += sign * PIECE_VAL[ptype]
            if ptype in ('c', 'p', 'm') and 2 <= x <= 6 and 2 <= y <= 7:
                score += sign * 16.0
            score += sign * _pawn_progress_bonus(ptype, piece[0], x, y)
    return score


def heuristic_value(board, side):
    return math.tanh(raw_side_score(board, side) / 1300.0)


def terminal_value(board_obj, side):
    enemy = opponent(side)
    if find_jiang(board_obj.board, side) is None:
        return -1.0
    if find_jiang(board_obj.board, enemy) is None:
        return 1.0
    if not get_all_legal(board_obj, side):
        return -1.0
    return None


def static_side_score(board, side):
    return raw_side_score(board, side)


# ── Node ──────────────────────────────────────────────────────────────────────

class PUCTNode:
    __slots__ = ('side', 'parent', 'move', 'prior',
                 'children', 'visit_count', 'value_sum', 'is_expanded')

    def __init__(self, side, parent=None, move=None, prior=0.0):
        self.side        = side
        self.parent      = parent
        self.move        = move
        self.prior       = prior
        self.children    = {}
        self.visit_count = 0
        self.value_sum   = 0.0
        self.is_expanded = False

    @property
    def q_value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def puct_score(self, c_puct=2.5):
        u = c_puct * self.prior * math.sqrt(max(1, self.parent.visit_count)) / (1 + self.visit_count)
        return -self.q_value + u

    def best_child(self, c_puct=2.5):
        return max(self.children.values(), key=lambda n: n.puct_score(c_puct))

    def expand(self, priors):
        next_side = 'r' if self.side == 'b' else 'b'
        for move, prob in priors.items():
            self.children[move] = PUCTNode(next_side, parent=self,
                                           move=move, prior=prob)
        self.is_expanded = True

    def add_dirichlet_noise(self, alpha=0.3, epsilon=0.25):
        if not self.children:
            return
        moves = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves))
        for m, n in zip(moves, noise):
            self.children[m].prior = (1 - epsilon) * self.children[m].prior + epsilon * n

    def backup(self, value):
        """value is from the perspective of *this* node's side."""
        self.visit_count += 1
        self.value_sum   += value
        if self.parent:
            # Parent is the opponent — flip the sign
            self.parent.backup(-value)


# ── MCTS ──────────────────────────────────────────────────────────────────────

class AlphaMCTS:

    def __init__(self, model, simulations=200, c_puct=2.0, device='cpu'):
        self.model       = model
        self.simulations = simulations
        self.c_puct      = c_puct
        self.device      = device
        self.model.eval()

    def _quick_move_score(self, board_obj, move, side):
        board = board_obj.board
        sx, sy, ex, ey = move
        piece = board[sy][sx]
        target = board[ey][ex]
        if piece == '.':
            return -MATE_SCORE

        enemy = opponent(side)
        moving_value = PIECE_VAL[piece[-1]]
        captured_value = PIECE_VAL[target[-1]] if target != '.' else 0
        source_attacked = is_attacked(board, sx, sy, enemy)

        captured = board_obj.move(sx, sy, ex, ey)
        try:
            if find_jiang(board_obj.board, enemy) is None:
                return MATE_SCORE

            score = raw_side_score(board_obj.board, side)
            if captured != '.':
                score += captured_value * 0.80 - moving_value * 0.04

            gives_check = is_in_check(board_obj.board, enemy)
            if gives_check:
                score += 170.0

            landing_attacked = is_attacked(board_obj.board, ex, ey, enemy)
            if landing_attacked:
                loss = max(0.0, moving_value - captured_value)
                score -= loss * 1.45
                if moving_value >= 400 and captured_value == 0:
                    score -= 220.0
                elif moving_value > captured_value:
                    score -= 60.0
            elif source_attacked and moving_value >= 350:
                score += min(260.0, moving_value * 0.45)

            if piece[-1] == 'z':
                score += _pawn_progress_bonus(piece[-1], side, ex, ey)
            return score
        finally:
            board_obj.undo_move(sx, sy, ex, ey, captured)

    def _best_tactical_score(self, board_obj, legal_moves, side,
                             include_quiet_checks=False):
        board = board_obj.board
        enemy = opponent(side)
        best = 0.0
        for move in legal_moves:
            sx, sy, ex, ey = move
            piece = board[sy][sx]
            target = board[ey][ex]
            if piece == '.':
                continue
            valuable_escape = (
                PIECE_VAL[piece[-1]] >= 350 and
                is_attacked(board, sx, sy, enemy)
            )
            tactical = target != '.' or valuable_escape
            if include_quiet_checks and not tactical:
                captured = board_obj.move(*move)
                try:
                    tactical = is_in_check(board_obj.board, enemy)
                finally:
                    board_obj.undo_move(sx, sy, ex, ey, captured)
            if not tactical:
                continue
            best = max(best, self._quick_move_score(board_obj, move, side))
        return best

    def _tactical_leaf_value(self, board_obj, legal_moves, side):
        enemy = opponent(side)
        base = raw_side_score(board_obj.board, side)
        own_tactic = self._best_tactical_score(
            board_obj, legal_moves, side, include_quiet_checks=True)
        enemy_moves = get_all_legal(board_obj, enemy)
        enemy_tactic = self._best_tactical_score(board_obj, enemy_moves, enemy)

        score = base + own_tactic * 0.35 - enemy_tactic * 0.45
        if is_in_check(board_obj.board, side):
            score -= 220.0
        if is_in_check(board_obj.board, enemy):
            score += 140.0
        return math.tanh(score / 1300.0)

    def _find_winning_moves(self, board_obj, legal_moves, side):
        enemy = opponent(side)
        wins = []
        for move in legal_moves:
            sx, sy, ex, ey = move
            captured = board_obj.move(sx, sy, ex, ey)
            try:
                if find_jiang(board_obj.board, enemy) is None:
                    wins.append(move)
                    continue
                if is_in_check(board_obj.board, enemy) and not get_all_legal(board_obj, enemy):
                    wins.append(move)
            finally:
                board_obj.undo_move(sx, sy, ex, ey, captured)
        return wins

    def _root_tactical_score(self, board_obj, move, side, prior=0.0, visits=0):
        board = board_obj.board
        sx, sy, ex, ey = move
        piece = board[sy][sx]
        target = board[ey][ex]
        moving_value = PIECE_VAL[piece[-1]]
        captured_value = PIECE_VAL[target[-1]] if target != '.' else 0
        enemy = opponent(side)

        captured = board_obj.move(sx, sy, ex, ey)
        try:
            terminal = terminal_value(board_obj, enemy)
            if terminal is not None and terminal < 0:
                return MATE_SCORE + visits

            score = static_side_score(board_obj.board, side)
            if captured != '.':
                score += min(180.0, captured_value * 0.25)
                score -= moving_value * 0.03
            if is_in_check(board_obj.board, enemy):
                score += 100.0

            worst_after_reply = score
            exposed_loss = 0.0
            for reply in get_all_legal(board_obj, enemy):
                rsx, rsy, rex, rey = reply
                reply_target = board_obj.board[rey][rex]
                tactical_reply = reply_target != '.' or (rex == ex and rey == ey)
                if not tactical_reply:
                    continue

                reply_capture = board_obj.move(rsx, rsy, rex, rey)
                try:
                    reply_terminal = terminal_value(board_obj, side)
                    if reply_terminal is not None and reply_terminal < 0:
                        reply_score = -MATE_SCORE
                    else:
                        reply_score = static_side_score(board_obj.board, side)
                        if is_in_check(board_obj.board, side):
                            reply_score -= 80.0
                    worst_after_reply = min(worst_after_reply, reply_score)
                finally:
                    board_obj.undo_move(rsx, rsy, rex, rey, reply_capture)

                if rex == ex and rey == ey:
                    exposed_loss = max(
                        exposed_loss,
                        max(0, moving_value - captured_value) * 1.45,
                    )

            return (
                worst_after_reply
                - exposed_loss
                + prior * 100.0
                + math.log1p(visits) * 8.0
            )
        finally:
            board_obj.undo_move(sx, sy, ex, ey, captured)

    def _guard_root_choice(self, board_obj, side, chosen, pi, counts,
                           tactical_scores=None):
        if not pi:
            return chosen, pi

        best_move = chosen
        best_score = -float('inf')
        chosen_score = None
        for move, prior in pi.items():
            if tactical_scores is not None and move in tactical_scores:
                score = tactical_scores[move]
            else:
                score = self._root_tactical_score(
                    board_obj, move, side, prior=prior,
                    visits=counts.get(move, 0))
            if move == chosen:
                chosen_score = score
            if score > best_score:
                best_score = score
                best_move = move

        if chosen_score is None:
            chosen_score = -float('inf')
        gap = best_score - chosen_score
        if best_move != chosen and (gap > 140.0 or (chosen_score < -220.0 and gap > 70.0)):
            adjusted = {m: p * 0.25 for m, p in pi.items()}
            adjusted[best_move] = adjusted.get(best_move, 0.0) + 0.75
            total = sum(adjusted.values()) or 1.0
            adjusted = {m: p / total for m, p in adjusted.items()}
            return best_move, adjusted

        return chosen, pi

    def _root_search_policy(self, board_obj, side, root, temperature):
        moves = list(root.children.keys())
        counts = {m: root.children[m].visit_count for m in moves}
        tactical_scores = {}
        logits = []
        for move in moves:
            node = root.children[move]
            root_q = -node.q_value if node.visit_count > 0 else 0.0
            tactical = self._root_tactical_score(
                board_obj, move, side, prior=node.prior,
                visits=node.visit_count,
            )
            tactical_scores[move] = tactical
            logits.append(
                math.log1p(node.visit_count) +
                0.35 * math.log(max(node.prior, 1e-8)) +
                1.15 * root_q +
                0.22 * max(-3.0, min(3.0, tactical / 650.0))
            )

        if not moves:
            return {}, counts, tactical_scores
        if temperature == 0:
            best_idx = max(range(len(moves)), key=lambda i: logits[i])
            return ({m: (1.0 if i == best_idx else 0.0)
                     for i, m in enumerate(moves)}, counts, tactical_scores)

        scale = 1.0 / max(0.25, float(temperature))
        arr = np.array(logits, dtype=np.float64) * scale
        arr -= arr.max()
        probs = np.exp(arr)
        probs /= probs.sum() or 1.0
        return ({m: float(p) for m, p in zip(moves, probs)},
                counts, tactical_scores)

    def _heuristic_priors(self, board_obj, legal_moves, side):
        board = board_obj.board
        scores = {}
        enemy = opponent(side)
        for move in legal_moves:
            sx, sy, ex, ey = move
            piece = board[sy][sx]
            target = board[ey][ex]
            score = 1.0
            moving_value = PIECE_VAL[piece[-1]]
            captured_value = PIECE_VAL[target[-1]] if target != '.' else 0
            source_attacked = is_attacked(board, sx, sy, enemy)
            if target != '.':
                score += captured_value / 120.0
                score -= moving_value / 1500.0
                if target[-1] == 'j':
                    score += 200.0
            if piece[-1] == 'z':
                if (side == 'b' and ey > sy) or (side == 'r' and ey < sy):
                    score += 0.4
            captured = board_obj.move(*move)
            try:
                gives_check = is_in_check(board_obj.board, enemy)
                if gives_check:
                    score += 4.0

                # Penalize risky landing squares cheaply. Generating every
                # opponent reply here makes neural self-play prohibitively slow.
                can_be_taken = is_attacked(board_obj.board, ex, ey, enemy)
                if can_be_taken:
                    loss = max(0, moving_value - captured_value)
                    score -= loss / 70.0
                    if moving_value >= 400 and captured_value == 0:
                        score -= 3.5
                    elif moving_value > captured_value:
                        score -= 0.8
                elif source_attacked and moving_value >= 350:
                    score += min(3.0, moving_value / 260.0)
            finally:
                board_obj.undo_move(sx, sy, ex, ey, captured)
            scores[move] = max(score, 0.05)
        total = sum(scores.values()) or 1.0
        return {m: v / total for m, v in scores.items()}

    # ── Single inference ──────────────────────────────────────────────────────

    def _infer(self, board_obj, legal_moves, side):
        """Returns (priors dict, value float) for the current board state."""
        tensor = board_to_tensor(board_obj.board, side).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            policy_logits, value = self.model(tensor)

        logits = policy_logits[0]
        idx    = torch.tensor([encode_move(*m) for m in legal_moves],
                              dtype=torch.long, device=self.device)
        probs = F.softmax(logits.index_select(0, idx), dim=0)
        prob_values = probs.detach().cpu().tolist()
        priors = {m: p for m, p in zip(legal_moves, prob_values)}
        tactical = self._heuristic_priors(board_obj, legal_moves, side)
        priors = {
            m: NETWORK_POLICY_WEIGHT * priors[m] +
               (1.0 - NETWORK_POLICY_WEIGHT) * tactical[m]
            for m in legal_moves
        }
        tactical_value = self._tactical_leaf_value(board_obj, legal_moves, side)
        mixed_value = (
            NETWORK_VALUE_WEIGHT * value.item() +
            (1.0 - NETWORK_VALUE_WEIGHT) * tactical_value
        )
        return priors, mixed_value

    # ── Single simulation (select → expand/evaluate → backup) ────────────────

    def _simulate(self, root, board_obj):
        node = root
        path = []  # list of (node, captured) for undo

        # ── Select: walk to leaf ──────────────────────────────────────────────
        while node.is_expanded and node.children:
            node = node.best_child(self.c_puct)
            captured = board_obj.move(*node.move)
            path.append((node, captured))

        # ── Evaluate leaf ─────────────────────────────────────────────────────
        terminal = terminal_value(board_obj, node.side)
        legal = [] if terminal is not None else get_all_legal(board_obj, node.side)

        if terminal is not None:
            # Terminal: current side has no moves → it loses
            value = terminal
        else:
            priors, raw_value = self._infer(board_obj, legal, node.side)
            if not node.is_expanded:
                node.expand(priors)
            # raw_value is from the network's "current side" perspective
            value = raw_value

        # ── Restore board ─────────────────────────────────────────────────────
        for n, captured in reversed(path):
            sx, sy, ex, ey = n.move
            board_obj.undo_move(sx, sy, ex, ey, captured)

        # ── Backup ───────────────────────────────────────────────────────────
        # value is from node.side's perspective; backup flips at each level
        node.backup(value)

    # ── Public interface ──────────────────────────────────────────────────────

    def get_move(self, board_obj, side, temperature=1.0, add_noise=False):
        if terminal_value(board_obj, side) is not None:
            return None, {}
        legal_moves = get_all_legal(board_obj, side)
        if not legal_moves:
            return None, {}

        winning_moves = self._find_winning_moves(board_obj, legal_moves, side)
        if winning_moves:
            scored = [
                (self._quick_move_score(board_obj, move, side), move)
                for move in winning_moves
            ]
            chosen = max(scored)[1]
            pi = {m: (1.0 if m == chosen else 0.0) for m in legal_moves}
            return chosen, pi

        root = PUCTNode(side)
        priors, _ = self._infer(board_obj, legal_moves, side)
        root.expand(priors)

        if add_noise:
            root.add_dirichlet_noise()

        for _ in range(self.simulations):
            self._simulate(root, board_obj)

        pi, counts, tactical_scores = self._root_search_policy(
            board_obj, side, root, temperature)

        moves  = list(pi.keys())
        p_arr  = np.array([pi[m] for m in moves], dtype=np.float64)
        p_arr /= p_arr.sum()
        chosen = moves[np.random.choice(len(moves), p=p_arr)]
        chosen, pi = self._guard_root_choice(
            board_obj, side, chosen, pi, counts, tactical_scores)
        return chosen, pi
