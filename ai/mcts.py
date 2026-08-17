"""无 rollout 的 PUCT 搜索（AlphaZero 风格）。

- select：沿 max Q + c_puct * P * sqrt(N_parent)/(1+n) 下钻到未展开叶子
- expand：叶子处网络一次评估得到 (p, v)，按"目标索引"聚合创建子节点
- 动作聚合：同一目标位置（place 落点 / eat 吃点）多个源棋子 → 取首个合法动作为代表，
  共享该目标索引的先验（与策略头输出维度一致，π 也按目标索引回传）
- backprop：value 每层翻转视角累计
"""
import math
from copy import deepcopy

import engine as E
from net import action_index, evaluate_net


class Node:
    __slots__ = ("state", "player", "parent", "action_idx", "action",
                 "children", "visits", "value_sum", "prior")

    def __init__(self, state, player, parent=None, action_idx=-1, action=None, prior=0.0):
        self.state = state
        self.player = player
        self.parent = parent
        self.action_idx = action_idx
        self.action = action
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior

    def value(self):
        return self.value_sum / self.visits if self.visits else 0.0


def other(player):
    return E.CROSS if player == E.CIRCLE else E.CIRCLE


def ucb(child, c_puct):
    q = child.value()
    prior = child.prior
    parent_visits = child.parent.visits
    return q + c_puct * prior * math.sqrt(parent_visits) / (1 + child.visits)


class MCTS:
    def __init__(self, net, c_puct=1.0):
        self.net = net
        self.c_puct = c_puct

    def search(self, state, player, budget):
        """从根状态搜 budget 次迭代，返回 (最佳动作, 根访问分布 π[目标索引], 根价值)。"""
        root = Node(state=deepcopy(state), player=player)
        for _ in range(budget):
            leaf = self._select(root)
            v = self._expand_or_eval(leaf)
            self._backprop(leaf, v)
        # 终选：访问最多的子节点（robust child）
        if not root.children:
            legal = E.legal_actions(root.state)
            return (legal[0] if legal else None), None, 0.0
        best = max(root.children, key=lambda c: c.visits)
        # π：按目标索引聚合访问数
        pi = [0.0] * (E.SIZE * E.SIZE * 2)
        total = 0
        for c in root.children:
            pi[c.action_idx] += c.visits
            total += c.visits
        if total:
            pi = [n / total for n in pi]
        return best.action, pi, root.value()

    def _select(self, root):
        node = root
        while node.children:
            node = max(node.children, key=lambda c: ucb(c, self.c_puct))
        return node

    def _expand_or_eval(self, node):
        """返回以 node.player 视角的价值。终局直接给 ±1，否则展开并网络评估。"""
        t = E.terminal(node.state)
        if t is not None:
            return 1.0 if t == node.player else (-1.0 if t != "draw" else 0.0)
        legal = E.legal_actions(node.state)
        if not legal:
            return 0.0  # 无子可走，视为和棋
        p, v = evaluate_net(self.net, node.state, node.player)
        # 按目标索引取代表动作
        per_target = {}
        for a in legal:
            idx = action_index(a)
            if idx not in per_target:
                per_target[idx] = a
        child_player = other(node.player)
        for idx, act in per_target.items():
            st = E.clone(node.state)
            E.apply(st, act)
            node.children.append(Node(state=st, player=child_player, parent=node,
                                      action_idx=idx, action=act, prior=p[idx]))
        return v

    def _backprop(self, leaf, v):
        node = leaf
        while node is not None:
            node.visits += 1
            node.value_sum += v
            v = -v
            node = node.parent
