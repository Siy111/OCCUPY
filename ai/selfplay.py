"""自对弈：生成 AlphaZero 训练样本 (棋盘平面, 合法mask, 目标分布π, 终局价值z)。

开局与主程序协议一致（mainX/mainO/subX/subO/lay/choose 回放构造）。
对局由当前网络引导 MCTS 双方下棋；终局/截断后按结算结果转成 z（当前玩家视角）。
"""
import random

import engine as E
import numpy as np
from net import build_input, legal_mask
from mcts import MCTS

SIZE = E.SIZE


def _city_spot(rng, occupied, min_dist):
    for _ in range(200):
        x = 12 + rng.randrange(SIZE - 24)
        y = 12 + rng.randrange(SIZE - 24)
        if all(abs(o[0] - x) + abs(o[1] - y) >= min_dist for o in occupied):
            return (x, y)
    raise RuntimeError("cannot place city")


def _near(c, off):
    x, y = c
    return [(x - off, y), (x + off, y), (x, y - off), (x, y + off)]


def make_start(seed):
    rng = random.Random(seed)
    main_x = (12 + rng.randrange(SIZE - 24), 12 + rng.randrange(SIZE - 24))
    main_o = _city_spot(rng, [main_x], 30)
    sub_x = _city_spot(rng, [main_x, main_o], 18)
    sub_o = _city_spot(rng, [main_x, main_o, sub_x], 18)

    open_actions = [
        {"opt": "mainX", "x": main_x[0], "y": main_x[1]},
        {"opt": "mainO", "x": main_o[0], "y": main_o[1]},
        {"opt": "subX", "x": sub_x[0], "y": sub_x[1]},
        {"opt": "subO", "x": sub_o[0], "y": sub_o[1]},
    ]
    for px, py in _near(main_x, 4):
        open_actions.append({"opt": "layX", "x": px, "y": py})
    for px, py in _near(main_o, 4):
        open_actions.append({"opt": "layO", "x": px, "y": py})
    resources = []
    for _ in range(10):
        x = 2 + rng.randrange(SIZE - 4)
        y = 2 + rng.randrange(SIZE - 4)
        if all(abs(r[0] - x) + abs(r[1] - y) >= 6 for r in resources):
            resources.append((x, y))
    if resources:
        open_actions.append({"opt": "resources", "resources": [{"x": x, "y": y} for x, y in resources]})
    open_actions.append({"opt": "choose", "x": main_x[0], "y": main_x[1], "player": "cross"})

    s = E.create_state()
    for a in open_actions:
        E.replay(s, a)
    return s


def generate_episode(net, budget, max_steps, seed):
    """一局自对弈 → (samples, winner)。samples 元素 = (planes, mask, pi, z)。"""
    mcts = MCTS(net)
    state = make_start(seed)
    samples = []
    winner = None
    for _ in range(max_steps):
        player = E.get_player(state)
        action, pi, _ = mcts.search(state, player, budget)
        if action is None:
            break  # 无子可走
        planes = build_input(state, player).squeeze(0).numpy()
        mask = legal_mask(state, player).numpy()
        samples.append({"planes": planes, "mask": mask, "pi": pi, "who": player})
        E.apply(state, action)
        t = E.terminal(state)
        if t is not None:
            winner = t
            break
    if winner is None:
        winner = E.settlement(state)["winner"]  # 截断 → 按结算胜负
    # z：当前玩家视角（+1 赢 / -1 输 / 0 平）
    for smp in samples:
        if winner == "draw":
            smp["z"] = 0.0
        else:
            smp["z"] = 1.0 if smp["who"] == E.PLAYER_MAP[winner] else -1.0
    return samples, winner


def generate_batch(net, budget, max_steps, n_games, base_seed=0):
    all_samples = []
    winners = []
    for g in range(n_games):
        samples, winner = generate_episode(net, budget, max_steps, base_seed + g)
        all_samples.extend(samples)
        winners.append(winner)
    return all_samples, winners
