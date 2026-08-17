"""迭代式自对弈训练（AlphaZero 闭环）：
自对弈（最佳网络引导 MCTS）→ 增量训练候选 → 对战评估 → 胜率>55% 接受替换 → 导出权重。

用法：python train.py [轮数] [每轮对局数] [epochs] [lr] [MCTS预算] [对战局数] [对局步数上限]
"""
import argparse
import copy
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

import engine as E
from net import OccupancyNet, N_ACTIONS, masked_policy, action_index, build_input, legal_mask
from mcts import MCTS
from selfplay import generate_episode, generate_batch_parallel, make_start


def device_of():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_batch(net, opt, samples, epochs, batch_size=64, device="cpu"):
    net.train()
    planes = torch.from_numpy(np.stack([s["planes"] for s in samples])).to(device)
    masks = torch.from_numpy(np.stack([s["mask"] for s in samples])).to(device)
    pis = torch.from_numpy(np.stack([s["pi"] for s in samples])).to(device)
    zs = torch.tensor([s["z"] for s in samples], dtype=torch.float32).to(device)
    n = len(samples)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            x = planes[idx]
            mask = masks[idx]
            pi = pis[idx]
            z = zs[idx]
            opt.zero_grad()
            logits, v = net(x)
            # value loss：MSE(v, z)
            loss_v = F.mse_loss(v.squeeze(1), z)
            # policy loss：合法目标上的交叉熵（非法 -inf 归零）
            lm = logits.clone()
            lm[mask == 0] = -1e9
            logp = F.log_softmax(lm, dim=-1)
            loss_p = -(pi * logp).sum(dim=-1).mean()
            loss = loss_v + loss_p
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        print(f"  epoch {ep + 1}/{epochs}: loss={total_loss / n:.4f} (v={loss_v.item():.4f} p={loss_p.item():.4f})")


def battle(net_a, net_b, n_games, budget, max_steps, device="cpu"):
    """net_a vs net_b，交替先手，返回 net_a 胜率（平局计 0.5）。"""
    mcts_a = MCTS(net_a)
    mcts_b = MCTS(net_b)
    wins_a = 0
    wins_b = 0
    draws = 0
    for g in range(n_games):
        a_is_cross = (g % 2 == 0)
        state = make_start(10000 + g)
        winner = None
        for _ in range(max_steps):
            player = E.get_player(state)
            mcts = mcts_a if (player == E.CROSS) == a_is_cross else mcts_b
            action, _, _ = mcts.search(state, player, budget)
            if action is None:
                break
            E.apply(state, action)
            t = E.terminal(state)
            if t is not None:
                winner = t
                break
        if winner is None:
            winner = E.settlement(state)["winner"]
        if winner == "draw":
            draws += 1
        elif (winner == "cross") == a_is_cross:
            wins_a += 1
        else:
            wins_b += 1
    return (wins_a + 0.5 * draws) / n_games, wins_a, wins_b, draws


def export_json(net, path, device="cpu"):
    """导出权重为 JSON（层列表），供 JS 端手工前向推理。"""
    sd = net.state_dict()
    layers = []
    for name in ["conv.0", "conv.2", "conv.4"]:
        layers.append({"name": "conv", "W": sd[name + ".weight"].cpu().tolist(),
                       "b": sd[name + ".bias"].cpu().tolist()})
    layers.append({"name": "fc", "W": sd["fc.weight"].cpu().tolist(),
                   "b": sd["fc.bias"].cpu().tolist()})
    layers.append({"name": "policy", "W": sd["policy.weight"].cpu().tolist(),
                   "b": sd["policy.bias"].cpu().tolist()})
    layers.append({"name": "value", "W": sd["value.weight"].cpu().tolist(),
                   "b": sd["value.bias"].cpu().tolist()})
    with open(path, "w") as f:
        json.dump({"channels": 5, "layers": layers}, f)
    print(f"model exported -> {path}")


def save_checkpoint(net, opt_state, r, path):
    """存档：最佳网络权重 + 训练器状态 + 当前轮数，供 --resume 精确续训。"""
    torch.save({"round": r, "best_state": net.state_dict(), "opt_state": opt_state}, path)
    print(f"checkpoint saved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rounds", type=int, nargs="?", default=3)
    ap.add_argument("games", type=int, nargs="?", default=16)
    ap.add_argument("epochs", type=int, nargs="?", default=10)
    ap.add_argument("lr", type=float, nargs="?", default=0.01)
    ap.add_argument("budget", type=int, nargs="?", default=50)
    ap.add_argument("battle_games", type=int, nargs="?", default=8)
    ap.add_argument("max_steps", type=int, nargs="?", default=120)
    ap.add_argument("out", type=str, nargs="?", default="model.json")
    ap.add_argument("--resume", type=str, default=None,
                    help="从 checkpoint 文件继续训练（从断点轮次的下一个轮次开始）")
    ap.add_argument("--workers", type=int, default=4,
                    help="自对弈并行进程数（默认 4；GPU 上自动退化为 1）")
    args = ap.parse_args()

    device = device_of()
    print(f"device: {device}")
    # checkpoint 统一放在 <out 同目录>/checkpoints/
    ckpt_dir = os.path.join(os.path.dirname(args.out) or ".", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best = OccupancyNet().to(device)
    have_best = False
    opt_state = None          # 上一轮训练器的状态，跨轮延续给下一轮候选
    start_round = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        best.load_state_dict(ckpt["best_state"])
        have_best = True
        opt_state = ckpt.get("opt_state")
        start_round = ckpt["round"] + 1
        print(f"resumed from {args.resume} (断点: round {ckpt['round']}), 从 round {start_round} 继续")

    for r in range(start_round, args.rounds + 1):
        print(f"--- round {r}/{args.rounds} ---")
        # 自对弈（第一轮用随机初始化网络引导，也成立）
        workers = 1 if device == "cuda" else args.workers
        samples, winners = generate_batch_parallel(
            best, args.budget, args.max_steps, args.games,
            base_seed=100000 + r * 1000, workers=workers)
        print(f"self-play {args.games} games -> {len(samples)} samples (workers={workers})")
        # 增量训练候选（从最佳权重继续）
        cand = copy.deepcopy(best) if have_best else OccupancyNet().to(device)
        if have_best:
            cand.load_state_dict(best.state_dict())
        cand_opt = torch.optim.SGD(cand.parameters(), lr=args.lr, momentum=0.9)
        if have_best and opt_state is not None:
            cand_opt.load_state_dict(opt_state)
        train_batch(cand, cand_opt, samples, args.epochs, device=device)
        # 对战评估
        if not have_best:
            print("round 1: 无条件接受（无对手）")
            best = cand
            have_best = True
        else:
            rate, wa, wb, dr = battle(cand, best, args.battle_games, args.budget, args.max_steps, device)
            print(f"battle candidate {wa}-{wb}-{dr} (win rate {rate:.2%})")
            if rate > 0.55:
                print("ACCEPTED (new best)")
                best = cand
            else:
                print("REJECTED (keep old best)")
        opt_state = cand_opt.state_dict()
        if have_best:
            export_json(best, args.out, device)
            save_checkpoint(best, opt_state, r, os.path.join(ckpt_dir, f"round_{r:03d}.pt"))


if __name__ == "__main__":
    main()
