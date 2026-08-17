# OCCUPY 纯网络 AI 对战程序（Python 版）：
# 复用训练侧同一套代码（engine/mcts/net），联网对局全程走"网络引导的纯 PUCT"，
# 先验 P 来自策略头、价值 Q 来自价值头，无任何手写走子规则。
#
# 用法: python bot.py <服务器地址> <房间ID> [MCTS预算] [模型路径]
#   例: python bot.py ws://127.0.0.1:8080 123456
#   例: python bot.py ws://127.0.0.1:8080 123456 100 model.json
import json
import random
import sys
import time

import websocket
import numpy as np

import engine as E
from net import OccupancyNet, load_weights_from_json
from mcts import MCTS

SIZE = E.SIZE


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [AI] {msg}", flush=True)


class PyBot:
    def __init__(self, ws_url, room_id, budget=100, model_path=None):
        self.ws_url = ws_url
        self.room_id = room_id
        self.budget = budget
        self.net = OccupancyNet()
        if model_path:
            load_weights_from_json(self.net, model_path)
            log(f"加载神经网络模型: {model_path}")
        else:
            log("警告：未提供模型，使用随机权重")
        self.mcts = MCTS(self.net)
        self.ws = None
        self.thinking = False
        self.retry_count = 0
        self.reset()

    def reset(self):
        self.history = []
        self.state = E.create_state()
        self.bot_side = None  # 'cross' | 'circle'（choose 后确定）
        self.role = None
        self.room_state = None
        self.surrendered = False
        self.game_end_reported = False

    # ---- 通信 ----
    def send(self, obj):
        if self.ws and self.ws.connected:
            self.ws.send(json.dumps(obj))

    def send_action(self, action):
        self.send({"type": "game_action", "action": action})

    # ---- 状态回放 ----
    def apply_action(self, state, action):
        picked = E.replay(state, action)
        if picked is not None:
            self.bot_side = "cross" if picked == E.CROSS else "circle"

    def rebuild(self):
        self.state = E.create_state()
        self.bot_side = None
        for a in self.history:
            self.apply_action(self.state, a)

    # ---- 选边（choose 阶段）----
    def do_choose(self):
        cross = E.clone(self.state)
        cross["buildPhase"] = "x-game"
        circle = E.clone(self.state)
        circle["buildPhase"] = "o-game"
        nc = len(E.legal_actions(cross))
        no = len(E.legal_actions(circle))
        picked = "cross" if nc > no else ("circle" if no > nc else ("cross" if random.random() < 0.5 else "circle"))
        pts = np.nonzero(self.state["grid"] == E.PLAYER_MAP[picked])
        if len(pts[0]) == 0:
            log("未找到可选的棋子，跳过选边")
            return
        pos = (int(pts[1][0]), int(pts[0][0]))  # (x, y)
        self.send_action({"opt": "choose", "x": pos[0], "y": pos[1]})
        self.bot_side = picked
        self.state["buildPhase"] = "x-game" if picked == "cross" else "o-game"
        self.history.append({"opt": "choose", "player": picked, "x": pos[0], "y": pos[1]})
        log(f"选边：执 {picked}（✕{nc} 着 / ◯{no} 着），对局期先手")
        self.maybe_act()

    # ---- 对局期走棋 ----
    def is_bot_turn(self):
        return (self.bot_side is not None
                and self.state["buildPhase"].endswith("game")
                and E.get_player(self.state) == E.PLAYER_MAP[self.bot_side])

    def check_surrender(self):
        if self.surrendered or not self.bot_side:
            return
        if any(c["owner"] == E.PLAYER_MAP[self.bot_side] and c["lost"] for c in self.state["mainCities"]):
            self.surrendered = True
            self.send({"type": "chat", "message": "AI 认输：主城已被攻破或围死"})
            log(f"房间 {self.room_id}: AI 认输")

    def handle_state_change(self):
        self.check_surrender()
        if self.surrendered:
            return
        if E.terminal(self.state) is not None and not self.game_end_reported:
            self.game_end_reported = True
            s = E.settlement(self.state)
            log(f"房间 {self.room_id}: 对局结束 ◯{s['circleTotal']} 分 vs ✕{s['crossTotal']} 分，胜者={s['winner']}")

    def report_attribution(self):
        s = E.settlement(self.state)
        log(f"房间 {self.room_id}: 归属 ◯残骸{s['circleKill']} 资源{s['circleResources']} 副城{s['circleSubCityScore']} "
            f"| ✕残骸{s['crossKill']} 资源{s['crossResources']} 副城{s['crossSubCityScore']}")

    def maybe_act(self):
        if self.surrendered or self.thinking or not self.is_bot_turn():
            return
        if not E.legal_actions(self.state):
            log(f"房间 {self.room_id}: 无棋可走，等待房主操作")
            return
        self.thinking = True
        if not self.is_bot_turn():
            self.thinking = False
            return
        t0 = time.time()
        player = E.PLAYER_MAP[self.bot_side]
        action, _, _, _ = self.mcts.search(self.state, player, self.budget)
        log(f"房间 {self.room_id}: 搜索耗时 {(time.time() - t0) * 1000:.0f}ms")
        if not action:
            self.thinking = False
            log(f"房间 {self.room_id}: 搜索无结果")
            return
        self.make_move(action)

    def make_move(self, action):
        opt = ("x-eat" if self.bot_side == "cross" else "o-eat") if action["type"] == "eat" \
            else ("x-place" if self.bot_side == "cross" else "o-place")
        self.send_action({"opt": "select", "x": action["selX"], "y": action["selY"]})
        # 停顿片刻再落子，模拟真人节奏（同时保持 thinking 防止重入）
        time.sleep(0.6 + random.random() * 0.6)
        self.send_action({"opt": opt, "fromX": action["selX"], "fromY": action["selY"], "x": action["x"], "y": action["y"]})
        act = {"opt": opt, "fromX": action["selX"], "fromY": action["selY"], "x": action["x"], "y": action["y"]}
        self.history.append(act)
        self.apply_action(self.state, act)
        self.thinking = False
        log(f"房间 {self.room_id}: 走棋 {opt} ({action['selX']},{action['selY']}) -> ({action['x']},{action['y']})")
        self.report_attribution()
        self.handle_state_change()

    # ---- 消息处理 ----
    def on_sync_state(self, data):
        self.history = data.get("history") or []
        self.rebuild()
        log(f"房间 {self.room_id}: 状态同步完成，历史 {len(self.history)} 步，AI 执 {self.bot_side or '未定'}")
        self.handle_state_change()
        if self.surrendered:
            return
        if self.state["buildPhase"] == "choose":
            self.do_choose()
        else:
            self.maybe_act()

    def on_sync_request(self):
        self.send({"type": "sync_response", "history": self.history, "isMyTurn": not self.is_bot_turn()})

    def on_game_action(self, action):
        if not action or not action.get("opt"):
            return
        opt = action["opt"]
        if opt == "undo":
            length = max(0, (action.get("historyLength") or 0) - 1)
            self.history = self.history[:length]
            self.rebuild()
            self.send_action({"opt": "undo-reply", "reply": True, "historyLength": action.get("historyLength")})
            log(f"房间 {self.room_id}: 同意悔棋，回退至 {length} 步")
        elif opt == "restart":
            self.send_action({"opt": "restart-reply", "reply": True})
            self.reset()
            log(f"房间 {self.room_id}: 同意重新开局")
        elif opt == "calculate":
            self.send_action({"opt": "calculate-reply", "reply": True})
        elif opt == "export":
            self.send_action({"opt": "export-reply", "reply": True})
        elif opt == "select":
            return  # 选中动作不改变状态
        else:
            self.history.append(action)
            self.apply_action(self.state, action)
            self.handle_state_change()
            if self.surrendered:
                return
            if self.state["buildPhase"] == "choose":
                self.do_choose()
            else:
                self.maybe_act()

    def handle_message(self, data):
        t = data.get("type")
        if t == "room_joined":
            self.room_state = "in"
            log(f"房间 {self.room_id}: 加入成功{'（房主）' if data.get('isHost') else '（客人）'}")
        elif t == "game_start":
            self.role = data.get("role")
            log(f"房间 {self.room_id}: 对局开始，角色 {self.role}")
        elif t == "sync_state":
            self.on_sync_state(data)
        elif t == "sync_request":
            self.on_sync_request()
        elif t == "game_action":
            self.on_game_action(data.get("action"))
        elif t == "room_closed":
            log(f"房间 {self.room_id}: 房间已关闭，AI 退出")
            sys.exit(0)
        elif t in ("player_left", "opponent_disconnected"):
            log(f"房间 {self.room_id}: 房主 {'离开' if t == 'player_left' else '断线'}")
        elif t == "player_rejoined":
            log(f"房间 {self.room_id}: 房主已重连")
        elif t == "error":
            log(f"房间 {self.room_id}: 服务器错误 {data.get('message', '')}")
            if "不存在" in (data.get("message") or ""):
                sys.exit(1)

    def connect(self):
        while True:
            try:
                self.ws = websocket.create_connection(self.ws_url, timeout=60)
                log(f"房间 {self.room_id}: 已连接 {self.ws_url}")
                self.send({"type": "join_room", "roomId": self.room_id})
                while True:
                    raw = self.ws.recv()
                    if not raw:
                        continue
                    try:
                        self.handle_message(json.loads(raw))
                    except Exception as e:
                        log(f"消息处理异常: {e}")
            except (websocket.WebSocketException, OSError) as e:
                log(f"连接断开: {e}，3 秒后重连")
                time.sleep(3)
                self.retry_count += 1
                if self.retry_count > 25:
                    log(f"房间 {self.room_id}: 重连次数超限，退出")
                    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python bot.py <服务器地址> <房间ID> [MCTS预算] [模型路径]")
        print("示例: python bot.py ws://127.0.0.1:8080 123456")
        sys.exit(1)
    url = sys.argv[1]
    room = sys.argv[2]
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    model = sys.argv[4] if len(sys.argv) > 4 else None
    log(f"启动：服务器 {url}，房间 {room}，MCTS预算={budget}，模型={model or '（随机权重）'}")
    bot = PyBot(url, room, budget, model)
    bot.connect()
