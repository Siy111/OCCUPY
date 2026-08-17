# 占城棋 AI

基于双头神经网络（策略 + 价值）+ 无 rollout 的 PUCT 搜索实现，包含**自对弈训练**和**联网对战**两个部分。

## 文件结构

| 文件 | 说明 |
| --- | --- |
| `engine.py` | 规则引擎。实现走子判定、围城、结算等 |
| `net.py` | 双头网络：5 通道棋盘输入，3 层卷积 + 全连接，输出 10368 维策略头 + 1 维价值头 |
| `mcts.py` | PUCT 搜索。叶子处由网络一次评估得到 (p, v)，先验取策略头、价值取价值头 |
| `selfplay.py` | 自对弈。随机生成开局，双方用网络引导 MCTS 下棋，产出 (planes, mask, π, z) 样本 |
| `train.py` | 训练主程序：自对弈 → 增量训练 → 与当前最佳对战 → 胜率达标则替换 → 导出权重 |
| `bot.py` | 联网对战程序，通过 WebSocket 协议加入房间下棋 |

## 环境要求

- Python 3.10+
- 依赖：`torch`、`numpy`、`websocket-client`

安装（CPU 版 torch）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy websocket-client
```

如需 GPU 版，参照 PyTorch 官网按 CUDA 版本安装。

## 训练

```bash
python ai/train.py [轮数] [每轮对局数] [epochs] [学习率] [MCTS预算] [对战局数] [步数上限] [输出文件] [--resume 检查点]
```

示例：

```bash
python ai/train.py 3 16 10 0.01 50 8 120 ai/model.json
```

参数说明：

| 参数 | 示例值 | 含义 |
| --- | --- | --- |
| 轮数（rounds） | 3 | 自对弈 → 训练 → 对战评估的完整循环重复几次 |
| 每轮对局数（games） | 16 | 每轮自对弈生成多少局棋，决定样本量 |
| epochs | 10 | 同一批样本被反复训练几遍，越大拟合越充分 |
| 学习率（lr） | 0.01 | 梯度下降的步长，太大不稳定、太小收敛慢 |
| MCTS预算（budget） | 50 | 每步棋搜索的模拟次数，越大棋力越强、越慢 |
| 对战局数（battle_games） | 8 | 候选网络与当前最佳对战几局来判定是否替换 |
| 步数上限（max_steps） | 120 | 单局步数超过该值强制结束，防止死循环拖垮训练 |
| 输出文件（out） | ai/model.json | 最佳权重导出路径 |
| `--resume` | ai/checkpoints/round_003.pt | 从该检查点继续训练，跳过已完成轮次 |
| `--workers` | 4 | 自对弈并行进程数，多核 CPU 上可显著提速；GPU 训练时自动退化为串行 |

训练流程（每轮）：

1. 用当前最佳网络引导自对弈，生成训练样本
2. 从当前最佳权重继续训练得到候选网络
3. 候选与最佳对战（交替先手），统计胜率
4. 胜率 > 0.55 则候选成为新最佳，否则丢弃
5. 每轮结束将最佳权重导出为 `model.json`，并将检查点存到 `ai/checkpoints/round_###.pt`

输出文件：

- `model.json`：最新最佳权重，每次导出覆盖，供 bot 加载
- `ai/checkpoints/round_001.pt`、`round_002.pt` …：每轮一份完整存档（网络权重 + 训练器状态 + 轮数），可回滚、可续训

续训示例（练到第 5 轮，从第 3 轮的存档继续）：

```bash
python ai/train.py 5 16 10 0.01 50 8 120 ai/model.json --resume ai/checkpoints/round_003.pt
```

## 推理

```bash
python ai/bot.py <服务器地址> <房间ID> [MCTS预算] [模型路径]
```

示例：

```bash
python ai/bot.py ws://127.0.0.1:8080 123456 100 ai/model.json
```

bot 以客人身份加入房间，建城阶段回放状态，choose 阶段选择合法动作更多的一方执子，对局期轮到己方时用 MCTS 搜索出招，主城被攻破或围死后自动认输。

## 模型文件

`model.json` 为推理权重，结构：

```
{"channels": 5, "layers": [conv×3, fc, policy, value]}
```

- conv×3：卷积层（stride 2，通道 16/32/32）
- fc：全连接 2592→128
- policy：128→10368（place 目标 5184 + eat 目标 5184）
- value：128→1（tanh 输出，当前视角胜率 [-1, 1]）

## 参考

- [AlphaZero Primer（Lc0）](http://pr184-draft.lc0.org/dev/lc0/search/alphazero/)：PUCT 公式与搜索流程的图解说明
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)：AlphaZero 原论文（PDF）
