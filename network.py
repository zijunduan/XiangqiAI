"""
network.py — 象棋神经网络（策略头 + 价值头）

输入：14 通道 × 10 × 9 的棋盘张量
  通道 0-6 : 黑方 7 种棋子的占位图（有=1，无=0）
  通道 7-13: 红方 7 种棋子的占位图
输出：
  policy : 长度 2314 的向量（所有可能着法的 logit）
           编码方式: from_pos * 9*10 + to_pos，超出范围的位置在解码时过滤
  value  : 标量，范围 [-1, 1]，正值表示黑方占优

依赖: torch >= 2.0
安装: pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─── 棋子编码 ──────────────────────────────────────────────────────────────
PIECE_ORDER = ['c', 'm', 'x', 's', 'j', 'p', 'z']   # 7 种，车马象士将炮兵
PIECE_CH_B = {p: i     for i, p in enumerate(PIECE_ORDER)}  # 黑方通道 0-6
PIECE_CH_R = {p: i + 7 for i, p in enumerate(PIECE_ORDER)}  # 红方通道 7-13
SIDE_CHANNEL = 14
N_CHANNELS = 15

# ─── 着法编解码 ────────────────────────────────────────────────────────────
# 用 from_square * 90 + to_square 编码，共 90*90=8100 个槽位
# 实际合法着法远少于此，非法位置的 logit 在采样时被 mask 掉
MOVE_DIM = 90 * 90   # 8100


def board_to_tensor(board, side=None):
    """
    将 10×9 字符串棋盘转为 (14, 10, 9) float32 张量。
    可批量使用：返回值直接 unsqueeze(0) 后送入网络。
    """
    t = np.zeros((N_CHANNELS, 10, 9), dtype=np.float32)
    for y in range(10):
        for x in range(9):
            p = board[y][x]
            if p == '.':
                continue
            piece_side, ptype = p[0], p[-1]
            ch = PIECE_CH_B[ptype] if piece_side == 'b' else PIECE_CH_R[ptype]
            t[ch, y, x] = 1.0
    if side == 'b':
        t[SIDE_CHANNEL, :, :] = 1.0
    elif side == 'r':
        t[SIDE_CHANNEL, :, :] = -1.0
    return torch.from_numpy(t)


def encode_move(sx, sy, ex, ey):
    """着法 → 整数索引"""
    from_sq = sy * 9 + sx
    to_sq   = ey * 9 + ex
    return from_sq * 90 + to_sq


def decode_move(idx):
    """整数索引 → (sx, sy, ex, ey)"""
    from_sq, to_sq = divmod(idx, 90)
    sy, sx = divmod(from_sq, 9)
    ey, ex = divmod(to_sq, 9)
    return sx, sy, ex, ey


def moves_to_mask(moves):
    """
    合法着法列表 → 长度 MOVE_DIM 的 bool 掩码张量（合法位置为 True）。
    用于在 policy 输出上 mask 掉非法着法。
    """
    mask = torch.zeros(MOVE_DIM, dtype=torch.bool)
    for sx, sy, ex, ey in moves:
        mask[encode_move(sx, sy, ex, ey)] = True
    return mask


# ─── 残差块 ────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


# ─── 主网络 ────────────────────────────────────────────────────────────────

class ChessNet(nn.Module):
    """
    小型 AlphaZero 风格网络，适合在 CPU 上训练。
    channels=64, n_res=6 在普通笔记本上单次前向约 5-10ms。

    若有 GPU，可将 channels 提高到 128，n_res 提高到 10。
    """

    def __init__(self, channels=64, n_res=6):
        super().__init__()
        # 主干：输入卷积 + 若干残差块
        self.stem = nn.Sequential(
            nn.Conv2d(N_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_res)])

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 10 * 9, MOVE_DIM),
        )

        # 价值头
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * 10 * 9, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),    # 输出 [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, 14, 10, 9)
        returns: policy_logits (B, MOVE_DIM), value (B, 1)
        """
        x = self.res_blocks(self.stem(x))
        return self.policy_head(x), self.value_head(x)

    @torch.no_grad()
    def predict(self, board, legal_moves, side=None, device='cpu'):
        """
        单局面推理接口，供 MCTS 调用。
        返回:
          probs: dict {(sx,sy,ex,ey): probability}  只含合法着法
          value: float，正值黑优
        """
        self.eval()
        tensor = board_to_tensor(board, side).unsqueeze(0).to(device)
        logits, value = self(tensor)

        logits = logits.squeeze(0)                    # (MOVE_DIM,)
        mask   = moves_to_mask(legal_moves).to(device)

        # 非法着法置为 -inf，softmax 后概率为 0
        logits[~mask] = float('-inf')
        probs_tensor = F.softmax(logits, dim=0)

        probs = {}
        for move in legal_moves:
            idx = encode_move(*move)
            probs[move] = probs_tensor[idx].item()

        return probs, value.item()


# ─── 存取 ──────────────────────────────────────────────────────────────────

def save_model(model, path='chess_model.pt'):
    torch.save(model.state_dict(), path)
    print(f"[网络] 已保存到 {path}")


def _load_state_dict_compatible(model, state):
    current = model.state_dict()
    adapted = {}
    skipped = []
    for key, value in state.items():
        if key not in current:
            skipped.append(key)
            continue
        target = current[key]
        if value.shape == target.shape:
            adapted[key] = value
            continue
        if key == 'stem.0.weight' and value.ndim == 4 and target.ndim == 4:
            if value.shape[0] == target.shape[0] and value.shape[2:] == target.shape[2:]:
                copied = target.clone()
                channels = min(value.shape[1], target.shape[1])
                copied[:, :channels, :, :] = value[:, :channels, :, :]
                adapted[key] = copied
                continue
        skipped.append(key)
    current.update(adapted)
    model.load_state_dict(current)
    return skipped


def load_model(path='chess_model.pt', device='cpu', **kwargs):
    model = ChessNet(**kwargs)
    state = torch.load(path, map_location=device)
    skipped = _load_state_dict_compatible(model, state)
    model.to(device)
    model.eval()
    if skipped:
        print(f"[network] compatible load skipped {len(skipped)} tensors")
    print(f"[网络] 已从 {path} 加载")
    return model
