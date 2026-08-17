// WebSocket服务器端
const WebSocket = require('ws');
const http = require('http');
const server = http.createServer();
const wss = new WebSocket.Server({ server });

// 房间表：roomId → Room
const rooms = new Map();

// 断线后保留槽位的重连宽限期（毫秒）
const RECONNECT_GRACE = 60000;

// 统一日志：格式 [时间] [级别] [上下文] 消息
function log(level, context, message) {
    const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    const ctx = context ? `[${context}] ` : '';
    const fn = level === 'ERROR' ? console.error : level === 'WARN' ? console.warn : console.log;
    fn(`[${time}] [${level}] ${ctx}${message}`);
}

// ==================== 房间模型 ====================
// 每个房间管理两名玩家（first_player 房主/先手，second_player 客人/后手）。
// 玩家槽位值：WebSocket（在线）| null（离线，宽限期内等待重连）。
// 连接对象上挂载 room/role 归属，断线定位从"遍历全表"降为 O(1)。
class Room {
    constructor(hostWs) {
        this.id = generateRoomId();
        this.players = new Map(); // role → ws | null
        this.isGameStarted = false;
        this.graceTimer = null;
        this.addPlayer('first_player', hostWs);
    }

    // ---- 槽位 ----

    addPlayer(role, ws) {
        this.players.set(role, ws);
        ws.room = this;
        ws.role = role;
    }

    onlineCount() {
        let count = 0;
        for (const ws of this.players.values()) if (ws) count++;
        return count;
    }

    // 找一个可用槽位：null（离线待重连）或已断开的僵尸连接（服务器尚未感知断线）。
    // 这是重连/加入的统一入口，僵尸连接在此释放，从结构上避免"占位导致房间已满"。
    availableSlot() {
        for (const [role, ws] of this.players) {
            if (ws === null || ws.readyState !== WebSocket.OPEN) {
                if (ws !== null) this.players.set(role, null); // 释放僵尸连接
                return role;
            }
        }
        return null;
    }

    // ---- 消息 ----

    send(ws, type, payload = {}) {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type, ...payload }));
    }

    broadcast(type, payload = {}, exceptWs = null) {
        const message = JSON.stringify({ type, ...payload });
        for (const ws of this.players.values()) {
            if (ws && ws !== exceptWs && ws.readyState === WebSocket.OPEN) ws.send(message);
        }
    }

    // ---- 断线 / 重连 / 生命周期 ----

    // 连接断开：释放槽位、通知对手、进入重连宽限期
    handleDisconnect(role, ws) {
        if (this.players.get(role) !== ws) return; // 槽位已被其他连接占用，忽略
        this.players.set(role, null);
        this.broadcast('opponent_disconnected', { message: '对手断线，等待重连...' });
        this.startGrace();
        log('INFO', `房间 ${this.id}`, `玩家 ${role} 断开，进入重连宽限期`);
    }

    // 重连：恢复槽位、清除宽限期、通知对手并请求同步
    reconnect(role, ws) {
        this.addPlayer(role, ws);
        if (this.graceTimer) { clearTimeout(this.graceTimer); this.graceTimer = null; }
        this.broadcast('player_rejoined', { message: '对手已重连' }, ws);
        this.broadcast('sync_request', { message: '对手重连，请发送游戏状态' }, ws);
    }

    // 启动宽限期：超时后按房主/客人/双方在线情况清理
    startGrace() {
        if (this.graceTimer) clearTimeout(this.graceTimer);
        this.graceTimer = setTimeout(() => {
            this.graceTimer = null;
            if (this.onlineCount() === 0) {
                this.close(); // 双方均未重连
            } else if (this.players.get('first_player') === null) {
                this.close('房主已断开，房间关闭'); // 房主未重连，关闭房间
            } else {
                this.players.delete('second_player'); // 客人未重连，释放槽位等待新玩家
                this.broadcast('player_left', { message: '对手已离开' }); // 房间保留，房主继续等待
                log('INFO', `房间 ${this.id}`, '客人断开超时，槽位已释放，等待新玩家');
            }
        }, RECONNECT_GRACE);
    }

    // 关闭房间并清理（message 存在时广播 room_closed，通知在场玩家退出对战）
    close(message = null, exceptWs = null) {
        if (this.graceTimer) { clearTimeout(this.graceTimer); this.graceTimer = null; }
        if (message) this.broadcast('room_closed', { message }, exceptWs);
        rooms.delete(this.id);
        log('INFO', `房间 ${this.id}`, `已删除${message ? `：${message}` : ''}`);
    }

    // 双方在线时开始游戏
    tryStart() {
        if (this.isGameStarted || this.onlineCount() !== 2) return;
        this.isGameStarted = true;
        this.send(this.players.get('first_player'), 'game_start', { role: 'first_player', message: '你是先手，请开始游戏' });
        this.send(this.players.get('second_player'), 'game_start', { role: 'second_player', message: '你是后手，等待先手操作' });
        log('INFO', `房间 ${this.id}`, '游戏开始');
    }
}

// ==================== 连接管理 ====================
wss.on('connection', (ws) => {
    log('INFO', '连接', `新连接建立（当前连接数 ${wss.clients.size}）`);
    ws.isAlive = true;
    ws.on('pong', () => { ws.isAlive = true; });

    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            const ctx = ws.room ? `房间 ${ws.room.id}` : '连接';
            log('INFO', ctx, `收到消息: ${data.type}`);
            handleMessage(ws, data);
        } catch (error) {
            log('WARN', '连接', `消息解析失败: ${error.message}`);
        }
    });

    ws.on('close', () => {
        const ctx = ws.room ? `房间 ${ws.room.id}` : '连接';
        log('INFO', ctx, `连接关闭（角色 ${ws.role || '未入房'}）`);
        // 通过连接上挂载的房间归属 O(1) 定位，无需遍历全部房间
        if (ws.room) ws.room.handleDisconnect(ws.role, ws);
    });

    ws.on('error', (error) => {
        log('ERROR', '连接', `连接错误: ${error.message}`);
    });
});

// 心跳保活：每 10 秒 ping 一次，未回 pong 的连接视为假死并断开
// 断开会触发 close → handleDisconnect，保证僵尸连接最终走正常断线流程（释放槽位、通知对手）
const heartbeatInterval = setInterval(() => {
    wss.clients.forEach((ws) => {
        if (ws.isAlive === false) return ws.terminate();
        ws.isAlive = false;
        ws.ping();
    });
}, 10000);

wss.on('close', () => clearInterval(heartbeatInterval));

// ==================== 消息处理 ====================
function handleMessage(ws, data) {
    switch (data.type) {
        case 'create_room': createRoom(ws); break;
        case 'join_room': joinRoom(ws, data); break;
        case 'close_room': closeRoom(ws); break;
        case 'leave_room': leaveRoom(ws); break;
        case 'game_action': broadcastGameAction(ws, data); break;
        case 'sync_response': handleSyncResponse(ws, data); break;
        case 'chat': handleChat(ws, data); break;
        default:
            const ctx = ws.room ? `房间 ${ws.room.id}` : '连接';
            log('WARN', ctx, `未知消息类型: ${data.type}`);
    }
}

// 聊天/认输消息：转发给房间内其他玩家（发送者不回显），附发送者角色
function handleChat(ws, data) {
    const room = ws.room;
    if (!room) {
        ws.send(JSON.stringify({ type: 'error', message: '聊天失败：未加入房间' }));
        return;
    }
    const text = String(data.message || '').slice(0, 200);
    log('INFO', `房间 ${room.id}`, `消息[${ws.role || '未知'}]: ${text}`);
    room.broadcast('chat', { message: text, from: ws.role || null }, ws);
}

// 创建房间（房主固定为先手）
function createRoom(ws) {
    const room = new Room(ws);
    rooms.set(room.id, room);
    room.send(ws, 'room_created', { roomId: room.id, isHost: true, message: '房间创建成功' });
    log('INFO', `房间 ${room.id}`, '创建成功');
}

// 加入房间（含断线重连）
function joinRoom(ws, data) {
    const room = rooms.get(data.roomId);
    if (!room) {
        ws.send(JSON.stringify({ type: 'error', message: '房间不存在' }));
        log('WARN', '连接', `拒绝加入：房间 ${data.roomId} 不存在`);
        return;
    }

    // 1) 有空槽（离线待重连 / 僵尸连接）：恢复原角色
    const slot = room.availableSlot();
    if (slot) {
        room.reconnect(slot, ws);
        room.send(ws, 'room_joined', { roomId: room.id, isHost: slot === 'first_player', role: slot, message: '重连成功' });
        log('INFO', `房间 ${room.id}`, `玩家 ${slot} 重连成功`);
        return;
    }

    // 2) 房间已空：重新加入作为房主
    if (room.players.size === 0) {
        room.addPlayer('first_player', ws);
        room.send(ws, 'room_joined', { roomId: room.id, isHost: true, message: '重新加入房间成功，等待对手' });
        log('INFO', `房间 ${room.id}`, '玩家重新加入作为房主');
        return;
    }

    // 3) 一个玩家在线：作为客人加入
    if (room.onlineCount() === 1) {
        room.addPlayer('second_player', ws);
        room.send(ws, 'room_joined', { roomId: room.id, isHost: false, message: '加入房间成功' });
        room.broadcast('player_joined', { message: '对手已加入' }, ws);
        room.broadcast('sync_request', { message: '请发送游戏状态' }, ws);
        log('INFO', `房间 ${room.id}`, '客人加入，请求同步');
        return;
    }

    // 4) 两人都在线
    ws.send(JSON.stringify({ type: 'error', message: '房间已满' }));
    log('WARN', `房间 ${room.id}`, '拒绝加入：房间已满');
}

// 关闭房间（仅房主可操作）
function closeRoom(ws) {
    const room = ws.room;
    if (room && ws.role === 'first_player') room.close('房主关闭了房间', ws);
}

// 退出房间（加入者操作）
function leaveRoom(ws) {
    const room = ws.room;
    if (!room) return;
    room.players.delete(ws.role); // 直接释放槽位（非断线，不进入宽限期）
    ws.room = null;
    if (room.onlineCount() === 0) {
        room.close();
    } else {
        room.broadcast('player_left', { message: '对手退出了房间' }); // 房间保留，房主继续等待
        log('INFO', `房间 ${room.id}`, '玩家退出');
    }
}

// 广播游戏操作
function broadcastGameAction(ws, data) {
    const room = ws.room;
    if (!room) return;
    room.broadcast('game_action', { action: data.action }, ws);
}

// 处理同步响应：向对方转发全量历史与回合状态；若游戏未开始则同步后开局
function handleSyncResponse(ws, data) {
    const room = ws.room;
    if (!room) return;
    room.broadcast('sync_state', { history: data.history, isMyTurn: data.isMyTurn }, ws);
    if (!room.isGameStarted) room.tryStart();
}

// ==================== 工具 ====================
// 生成房间ID
function generateRoomId() {
    let id;
    do {
        id = String(Math.floor(100000 + Math.random() * 900000));
    } while (rooms.has(id));
    return id;
}

// ==================== 启动 ====================
const PORT = 8080;
server.listen(PORT, '0.0.0.0', () => {
    log('INFO', null, `WebSocket服务器启动在端口 ${PORT}（本地: ws://localhost:${PORT}）`);
});