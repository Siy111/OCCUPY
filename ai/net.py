"""双头网络：棋盘平面输入（当前玩家视角）+ CNN 共享层 → 策略头 + 价值头。

策略头输出 N_ACTIONS = 72*72(place 目标) + 72*72(eat 目标) = 10368 维，
对合法目标 mask 后 softmax（AlphaZero 风格，动作空间固定、非法目标 -inf）。
价值头输出当前玩家视角的胜率估计（tanh ∈ [-1, 1]）。
"""
import torch
import torch.nn as nn
import numpy as np

import engine as E

SIZE = E.SIZE
N_PLACE = SIZE * SIZE
N_EAT = SIZE * SIZE
N_ACTIONS = N_PLACE + N_EAT


def action_index(act):
    """动作 → 策略索引（place 目标在前，eat 目标在后）。"""
    if act["type"] == "place":
        return act["y"] * SIZE + act["x"]
    return N_PLACE + act["y"] * SIZE + act["x"]


def build_input(state, player):
    """棋盘平面：5 通道 × 72×72（当前玩家视角）。"""
    grid = state["grid"]
    my = np.zeros((SIZE, SIZE), dtype=np.float32)
    enemy = np.zeros((SIZE, SIZE), dtype=np.float32)
    debris = np.zeros((SIZE, SIZE), dtype=np.float32)
    my_area = np.zeros((SIZE, SIZE), dtype=np.float32)
    en_area = np.zeros((SIZE, SIZE), dtype=np.float32)
    enemy_p = E.CIRCLE if player == E.CROSS else E.CROSS

    for y in range(SIZE):
        for x in range(SIZE):
            t = grid[y, x]
            if t == player:
                my[y, x] = 1.0
            elif t == enemy_p:
                enemy[y, x] = 1.0
            elif t in (E.MIX, E.RESOURCE):
                debris[y, x] = 1.0
            if E.own_area(state, player, x, y):
                my_area[y, x] = 1.0
            if E.enemy_area(state, player, x, y):
                en_area[y, x] = 1.0

    planes = np.stack([my, enemy, debris, my_area, en_area])
    return torch.from_numpy(planes).unsqueeze(0)  # (1, C, 72, 72)


def legal_mask(state, player):
    """合法目标 mask：对每个策略索引标记是否存在合法动作。"""
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    for act in E.legal_actions(state):
        mask[action_index(act)] = 1.0
    return torch.from_numpy(mask)


class OccupancyNet(nn.Module):
    def __init__(self, channels=5, hidden=128):
        super().__init__()
        # 72 -> 36 -> 18 -> 9（三次 stride-2 卷积快速降维）
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(9 * 9 * 32, hidden)
        self.fc_relu = nn.ReLU()
        self.policy = nn.Linear(hidden, N_ACTIONS)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        h = self.fc_relu(self.fc(h))
        logits = self.policy(h)
        value = torch.tanh(self.value(h))
        return logits, value


def masked_policy(logits, mask):
    """合法目标上的 softmax；非法目标置 -inf（全非法时退化为均匀）。"""
    m = mask == 0
    logits = logits.clone()
    logits[m] = -1e9
    if (mask == 0).all():
        return mask  # 无合法动作（防御）
    return torch.softmax(logits, dim=-1)


@torch.no_grad()
def evaluate_net(net, state, player):
    """MCTS 叶子评估：返回 (策略先验 np.array, 价值 float)。"""
    net.eval()
    x = build_input(state, player)
    mask = legal_mask(state, player)
    logits, value = net(x)
    p = masked_policy(logits.squeeze(0), mask)
    return p.numpy(), float(value.item())


def load_weights_from_json(net, path):
    """从 train.py 导出的 model.json 还原权重（conv.0/conv.2/conv.4 + fc + policy + value）。"""
    import json
    with open(path) as f:
        m = json.load(f)
    sd = {}
    names = ["conv.0", "conv.2", "conv.4"]
    for name, layer in zip(names, m["layers"][:3]):
        sd[name + ".weight"] = torch.tensor(layer["W"])
        sd[name + ".bias"] = torch.tensor(layer["b"])
    fc = m["layers"][3]
    sd["fc.weight"] = torch.tensor(fc["W"])
    sd["fc.bias"] = torch.tensor(fc["b"])
    pol = m["layers"][4]
    sd["policy.weight"] = torch.tensor(pol["W"])
    sd["policy.bias"] = torch.tensor(pol["b"])
    val = m["layers"][5]
    sd["value.weight"] = torch.tensor(val["W"])
    sd["value.bias"] = torch.tensor(val["b"])
    net.load_state_dict(sd)
    return net
