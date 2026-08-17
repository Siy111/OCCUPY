"""OCCUPY 规则引擎（Python 版）：与 ai/rules.js 严格同口径。

棋盘用 numpy int8 数组表示：0=empty 1=cross 2=circle 3=mix 4=resource。
城池状态用 dict 列表（与 JS 对象结构一致）。
"""
import numpy as np
from collections import deque

try:
    from scipy import ndimage as _ndimage
except ImportError:
    _ndimage = None

SIZE = 72
RANGE_PLACE = 2
RANGE_EAT = 1
OFFSETS_3x3 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
MAIN_INNER = 1
MAIN_OUTER = 3
SUB_AREA = 2
MAIN_SIZE = 9
SUB_SIZE = 7

EMPTY = 0
CROSS = 1
CIRCLE = 2
MIX = 3
RESOURCE = 4

PLAYER_MAP = {"cross": CROSS, "circle": CIRCLE}

# 边界种子环（8 邻域洪泛起点，与 rules.js 参考实现一致，含近角格）：
# x∈[1,SIZE-2]（y=1 或 SIZE-2）与 y∈[1,SIZE-2]（x=1 或 SIZE-2）
_RING_MASK = np.zeros((SIZE, SIZE), dtype=bool)
_RING_MASK[1:SIZE - 1, 1] = True
_RING_MASK[1:SIZE - 1, SIZE - 2] = True
_RING_MASK[1, 1:SIZE - 1] = True
_RING_MASK[SIZE - 2, 1:SIZE - 1] = True
_CONN8 = np.ones((3, 3), dtype=bool)


def create_state():
    return {
        "buildPhase": "x-main",
        "grid": np.zeros((SIZE, SIZE), dtype=np.int8),
        "mainCities": [],
        "subCities": [],
    }


def clone(state):
    return {
        "buildPhase": state["buildPhase"],
        "grid": state["grid"].copy(),
        "mainCities": [dict(c) for c in state["mainCities"]],
        "subCities": [dict(c) for c in state["subCities"]],
    }


# ---- 基础几何 ----
def get_player(state):
    return CROSS if state["buildPhase"].startswith("x-") else CIRCLE


def is_in_range(x, y, tx, ty, r):
    return abs(x - tx) <= r and abs(y - ty) <= r


def in_city(x, y, c, off):
    return is_in_range(x, y, c["x"], c["y"], off)


def inner_main(state, x, y):
    return any(in_city(x, y, c, MAIN_INNER) for c in state["mainCities"])


def sub_city(state, x, y):
    return any(in_city(x, y, c, SUB_AREA) for c in state["subCities"])


def own_area(state, player, x, y):
    for c in state["mainCities"]:
        if c["owner"] == player and in_city(x, y, c, MAIN_INNER if c["lost"] else MAIN_OUTER):
            return True
    for c in state["subCities"]:
        if c["occupied"] == player and not c["lost"] and in_city(x, y, c, SUB_AREA):
            return True
    return False


def enemy_area(state, player, x, y):
    for c in state["mainCities"]:
        if c["owner"] != player and not c["lost"] and in_city(x, y, c, MAIN_INNER if c["attacked"] else MAIN_OUTER):
            return True
    for c in state["subCities"]:
        if c["occupied"] and c["occupied"] != player and not c["attacked"] and in_city(x, y, c, SUB_AREA):
            return True
    return False


# ---- 走子判定 ----
def can_place(state, player, x, y):
    if inner_main(state, x, y):
        return False
    if enemy_area(state, player, x, y):
        return False
    return True


def is_path_clear(state, player, sx, sy, x, y):
    dx = x - sx
    dy = y - sy
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return False
    step_x = dx / steps
    step_y = dy / steps
    cx = float(sx)
    cy = float(sy)
    for i in range(1, steps):
        cx += step_x
        cy += step_y
        x_min = int(np.floor(cx))
        x_max = int(np.ceil(cx))
        y_min = int(np.floor(cy))
        y_max = int(np.ceil(cy))
        for rx in range(x_min, x_max + 1):
            for ry in range(y_min, y_max + 1):
                if rx < 0 or rx >= SIZE or ry < 0 or ry >= SIZE:
                    return False
                if state["grid"][ry, rx] != EMPTY:
                    return False
                if enemy_area(state, player, rx, ry):
                    return False
                if inner_main(state, rx, ry):
                    return False
    return True


def has_advantage(state, player, sx, sy):
    enemy = CIRCLE if player == CROSS else CROSS
    self_count = 0
    en_count = 0
    for dx, dy in OFFSETS_3x3:
        nx = sx + dx
        ny = sy + dy
        if nx < 0 or nx >= SIZE or ny < 0 or ny >= SIZE:
            continue
        t = state["grid"][ny, nx]
        if t == player:
            self_count += 1
        elif t == enemy:
            en_count += 1
        elif t == EMPTY and own_area(state, player, nx, ny):
            self_count += 1
        if self_count > 4:
            return True
        if en_count > 4:
            return False
    return self_count > en_count


def can_place_to(state, player, sx, sy, x, y):
    return (state["grid"][y, x] == EMPTY
            and can_place(state, player, x, y)
            and is_in_range(x, y, sx, sy, RANGE_PLACE)
            and is_path_clear(state, player, sx, sy, x, y))


def can_eat_at(state, player, sx, sy, x, y):
    t = state["grid"][y, x]
    return (t == CIRCLE or t == CROSS) and t != player \
        and is_in_range(x, y, sx, sy, RANGE_EAT) \
        and has_advantage(state, player, sx, sy)


# ---- 城池状态机 ----
def check_attack_condition(state, x, y, size, attacker):
    offset = size // 2
    count = 0
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        for i in range(-(offset - 1), offset):
            nx = x + i if dx == 0 else x + dx * offset
            ny = y + i if dy == 0 else y + dy * offset
            if nx < 0 or nx >= SIZE or ny < 0 or ny >= SIZE:
                continue
            if state["grid"][ny, nx] == attacker:
                count += 1
                break
    return count >= 4


def build_escape_grid(state, player):
    """可逃生区域（反向 flood fill）。返回 numpy int8 扁平数组：3=可达。

    与 rules.js 同口径。有 scipy 时用连通域标记一次成型（O(n)）；
    否则回退到 8 邻域向量化逐轮扩散。种子环为预计算常量，避免逐格 Python 循环。
    """
    g = np.zeros(SIZE * SIZE, dtype=np.int8)

    def mark_rect_v(cx, cy, off, v):
        xa = max(0, cx - off)
        xb = min(SIZE - 1, cx + off)
        ya = max(0, cy - off)
        yb = min(SIZE - 1, cy + off)
        g2[ya:(yb + 1), xa:(xb + 1)] = v

    # 先用 2D 视图做矩形填充，最后 flatten 回扁平数组
    g2 = g.reshape(SIZE, SIZE)
    for c in state["mainCities"]:
        owner_friendly = c["owner"] == player
        off = MAIN_INNER if (owner_friendly and c["lost"]) or (not owner_friendly and c["attacked"]) else MAIN_OUTER
        mark_rect_v(c["x"], c["y"], off, 1 if owner_friendly else 2)
    for c in state["subCities"]:
        if c["occupied"] == player and not c["lost"]:
            mark_rect_v(c["x"], c["y"], SUB_AREA, 1)
        elif c["occupied"] and c["occupied"] != player and not c["attacked"]:
            mark_rect_v(c["x"], c["y"], SUB_AREA, 2)

    grid = state["grid"]
    # 可通行掩码（与 can_pass 同口径）：己方棋子 / 己方区域(1) 总可过；敌区(2) 不可过；否则空才可过
    t = grid
    passable = (t == player) | (g2 == 1) | ((g2 != 2) & (t == EMPTY))
    # 洪泛有效区域 = 可通行 ∩ 非敌区（与原扩散中 grown &= passable + grown[g2==2]=False 等价）
    allowed = passable & (g2 != 2)
    # 边界种子：种子环上可通行且非敌区的格（向量化，替代逐格循环）
    seeds = allowed & _RING_MASK

    if _ndimage is not None:
        # 连通域一次成型：标记 allowed 的 8 连通分量，含种子的分量即为可逃生区域
        labels, _ = _ndimage.label(allowed, structure=_CONN8)
        seed_labels = np.unique(labels[seeds])
        seed_labels = seed_labels[seed_labels != 0]
        out = g2.copy()
        mark = np.isin(labels, seed_labels)
        # 与回退扩散一致：最外圈（0/SIZE-1）永不标记（围城判定只读城池所在格，外圈无影响）
        mark[0, :] = False
        mark[SIZE - 1, :] = False
        mark[:, 0] = False
        mark[:, SIZE - 1] = False
        out[mark] = 3
        return out.flatten()

    # 无 scipy 回退：8 邻域向量化扩散，直到不再变化
    reach = seeds.copy()
    while True:
        grown = reach.copy()
        for dy, dx in DIRS:
            grown[1:SIZE - 1, 1:SIZE - 1] |= reach[1 - dy:SIZE - 1 - dy, 1 - dx:SIZE - 1 - dx]
        grown &= allowed
        grown &= ~reach
        if not grown.any():
            break
        reach |= grown

    out = g2.copy()
    out[reach] = 3
    return out.flatten()


def check_city_surrounded(state):
    at_edge = lambda c: c["x"] == 0 or c["x"] == SIZE - 1 or c["y"] == 0 or c["y"] == SIZE - 1
    need_cross = False
    need_circle = False
    for c in state["mainCities"]:
        if not c["lost"] and not at_edge(c):
            if c["owner"] == CROSS:
                need_cross = True
            else:
                need_circle = True
    for c in state["subCities"]:
        if c["occupied"] and not c["lost"] and not at_edge(c):
            if c["occupied"] == CROSS:
                need_cross = True
            else:
                need_circle = True
    if not need_cross and not need_circle:
        return
    free_cross = build_escape_grid(state, CROSS) if need_cross else None
    free_circle = build_escape_grid(state, CIRCLE) if need_circle else None
    for c in state["mainCities"]:
        if not c["lost"] and not at_edge(c):
            f = free_cross if c["owner"] == CROSS else free_circle
            if f[c["y"] * SIZE + c["x"]] != 3:
                c["attacked"] = True
                c["lost"] = True
    for c in state["subCities"]:
        if c["occupied"] and not c["lost"] and not at_edge(c):
            f = free_cross if c["occupied"] == CROSS else free_circle
            if f[c["y"] * SIZE + c["x"]] != 3:
                c["attacked"] = True
                c["lost"] = True


def update_city_attacked(state, x, y):
    for c in state["mainCities"]:
        if c["lost"]:
            continue
        if is_in_range(x, y, c["x"], c["y"], MAIN_SIZE // 2):
            atk = CIRCLE if c["owner"] == CROSS else CROSS
            c["attacked"] = check_attack_condition(state, c["x"], c["y"], MAIN_SIZE, atk)
    for c in state["subCities"]:
        if c["lost"] or not c["occupied"]:
            continue
        if is_in_range(x, y, c["x"], c["y"], SUB_SIZE // 2):
            atk = CIRCLE if c["occupied"] == CROSS else CROSS
            c["attacked"] = check_attack_condition(state, c["x"], c["y"], SUB_SIZE, atk)


def check_sub_city_occupy(state, player, x, y):
    for c in state["subCities"]:
        if in_city(x, y, c, SUB_AREA) and not c["occupied"] and not c["lost"]:
            c["occupied"] = player


def check_city_attack(state, player, x, y):
    update_city_attacked(state, x, y)
    for c in state["mainCities"]:
        if c["owner"] != player and c["attacked"] and not c["lost"] and in_city(x, y, c, MAIN_OUTER):
            c["lost"] = True
    for c in state["subCities"]:
        if c["occupied"] != player and c["attacked"] and not c["lost"] and in_city(x, y, c, SUB_AREA):
            c["lost"] = True
            c["occupied"] = None
            c["attacked"] = False
    check_city_surrounded(state)


# ---- 对局期动作 ----
def legal_actions(state):
    """返回动作列表：[{'selX','selY','type','x','y'}]，与 rules.js 顺序无关（集合语义）。"""
    player = get_player(state)
    moves = []
    grid = state["grid"]
    ys, xs = np.nonzero(grid == player)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        for dy in range(-RANGE_PLACE, RANGE_PLACE + 1):
            for dx in range(-RANGE_PLACE, RANGE_PLACE + 1):
                if dx == 0 and dy == 0:
                    continue
                nx = sx + dx
                ny = sy + dy
                if nx < 0 or nx >= SIZE or ny < 0 or ny >= SIZE:
                    continue
                if grid[ny, nx] != EMPTY:
                    continue
                if can_place_to(state, player, sx, sy, nx, ny):
                    moves.append({"selX": sx, "selY": sy, "type": "place", "x": nx, "y": ny})
        for dy in range(-RANGE_EAT, RANGE_EAT + 1):
            for dx in range(-RANGE_EAT, RANGE_EAT + 1):
                if dx == 0 and dy == 0:
                    continue
                nx = sx + dx
                ny = sy + dy
                if nx < 0 or nx >= SIZE or ny < 0 or ny >= SIZE:
                    continue
                if can_eat_at(state, player, sx, sy, nx, ny):
                    moves.append({"selX": sx, "selY": sy, "type": "eat", "x": nx, "y": ny})
    return moves


def apply(state, action):
    player = get_player(state)
    if action["type"] == "place":
        state["grid"][action["y"], action["x"]] = player
        check_sub_city_occupy(state, player, action["x"], action["y"])
        check_city_attack(state, player, action["x"], action["y"])
    else:
        state["grid"][action["y"], action["x"]] = MIX
        update_city_attacked(state, action["x"], action["y"])
        check_city_surrounded(state)
    state["buildPhase"] = "o-game" if player == CROSS else "x-game"


def terminal(state):
    """终局判定：主城失守。统一返回字符串 "circle"/"cross"/"draw"（与 settlement 同口径）。"""
    c_lost = any(c["owner"] == CIRCLE and c["lost"] for c in state["mainCities"])
    x_lost = any(c["owner"] == CROSS and c["lost"] for c in state["mainCities"])
    if c_lost and x_lost:
        return "draw"
    if c_lost:
        return "circle"
    if x_lost:
        return "cross"
    return None


# ---- 结算（与 rules.js settlement 同口径）----
def settlement(state):
    circle_kill = 0
    cross_kill = 0
    circle_res = 0
    cross_res = 0
    circle_sub = 0
    cross_sub = 0
    grid = state["grid"]
    for y in range(SIZE):
        for x in range(SIZE):
            t = grid[y, x]
            if t != MIX and t != RESOURCE:
                continue
            circle_n = 0
            cross_n = 0
            for dx, dy in OFFSETS_3x3:
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= SIZE or ny < 0 or ny >= SIZE:
                    continue
                tp = grid[ny, nx]
                if tp == CIRCLE:
                    circle_n += 1
                elif tp == CROSS:
                    cross_n += 1
            if circle_n > cross_n:
                if t == MIX:
                    circle_kill += 1
                else:
                    circle_res += 1
            elif cross_n > circle_n:
                if t == MIX:
                    cross_kill += 1
                else:
                    cross_res += 1
    for c in state["subCities"]:
        if c["lost"]:
            continue
        owner = c["occupied"] if c["occupied"] else c["owner"]
        friendly = 0
        for dy in range(-SUB_AREA, SUB_AREA + 1):
            for dx in range(-SUB_AREA, SUB_AREA + 1):
                nx = c["x"] + dx
                ny = c["y"] + dy
                if nx < 0 or nx >= SIZE or ny < 0 or ny >= SIZE:
                    continue
                tp = grid[ny, nx]
                if (owner == CIRCLE and tp == CIRCLE) or (owner == CROSS and tp == CROSS):
                    friendly += 1
        if owner == CIRCLE:
            circle_sub += friendly
        else:
            cross_sub += friendly
    circle_total = circle_kill + circle_res + circle_sub
    cross_total = cross_kill + cross_res + cross_sub
    circle_main_lost = any(c["owner"] == CIRCLE and c["lost"] for c in state["mainCities"])
    cross_main_lost = any(c["owner"] == CROSS and c["lost"] for c in state["mainCities"])
    if not circle_main_lost and cross_main_lost:
        winner = "circle"
    elif circle_main_lost and not cross_main_lost:
        winner = "cross"
    elif circle_total > cross_total:
        winner = "circle"
    elif cross_total > circle_total:
        winner = "cross"
    else:
        winner = "draw"
    return {
        "circleKill": circle_kill, "crossKill": cross_kill,
        "circleResources": circle_res, "crossResources": cross_res,
        "circleSubCityScore": circle_sub, "crossSubCityScore": cross_sub,
        "circleTotal": circle_total, "crossTotal": cross_total,
        "winner": winner,
    }


def state_signature(state):
    """对局终态/中间态签名（供 JS-Python 对齐验证用）。"""
    return {
        "buildPhase": state["buildPhase"],
        "grid": state["grid"].tolist(),
        "mainCities": [dict(c) for c in state["mainCities"]],
        "subCities": [dict(c) for c in state["subCities"]],
    }


# ---- 协议历史动作回放（与 rules.js replay 同口径，用于构造开局/回放历史）----
def replay(state, action):
    opt = action["opt"]
    if opt == "mainX":
        state["mainCities"].append({"x": action["x"], "y": action["y"], "owner": CROSS, "attacked": False, "lost": False})
        state["buildPhase"] = "o-main"
    elif opt == "mainO":
        state["mainCities"].append({"x": action["x"], "y": action["y"], "owner": CIRCLE, "attacked": False, "lost": False})
        state["buildPhase"] = "x-sub"
    elif opt == "subX":
        state["subCities"].append({"x": action["x"], "y": action["y"], "owner": CROSS, "occupied": None, "attacked": False, "lost": False})
        state["buildPhase"] = "o-sub"
    elif opt == "subO":
        state["subCities"].append({"x": action["x"], "y": action["y"], "owner": CIRCLE, "occupied": None, "attacked": False, "lost": False})
        state["buildPhase"] = "x-lay"
    elif opt == "layX":
        state["grid"][action["y"], action["x"]] = CROSS
        state["buildPhase"] = "o-lay"
    elif opt == "layO":
        state["grid"][action["y"], action["x"]] = CIRCLE
        state["buildPhase"] = "choose"
    elif opt == "choose":
        picked = action.get("player")
        if picked is not None and isinstance(picked, str):
            picked = PLAYER_MAP[picked]
        if picked is None:
            picked = state["grid"][action["y"], action["x"]]
        state["buildPhase"] = "x-game" if picked == CROSS else "o-game"
        return picked
    elif opt in ("x-place", "o-place"):
        apply(state, {"type": "place", "x": action["x"], "y": action["y"]})
    elif opt in ("x-eat", "o-eat"):
        apply(state, {"type": "eat", "x": action["x"], "y": action["y"]})
    elif opt == "res":
        state["grid"][action["y"], action["x"]] = RESOURCE
    elif opt == "eptRes":
        state["grid"][action["y"], action["x"]] = EMPTY
    elif opt == "resources":
        for r in action.get("resources", []):
            state["grid"][r["y"], r["x"]] = RESOURCE
    return None
