// WebSocket服务器端
const WebSocket = require('ws');
const http = require('http');
const server = http.createServer();
const wss = new WebSocket.Server({ server });

// 房间管理
const rooms = new Map();

// 连接管理
wss.on('connection', (ws) => {
    console.log('新连接');
    
    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            console.log('服务器收到消息:', data.type);
            console.log('消息内容:', data);
            handleMessage(ws, data);
        } catch (error) {
            console.error('消息解析错误:', error);
        }
    });
    
    ws.on('close', () => {
        console.log('连接关闭');
        // 清理房间
        for (const [roomId, room] of rooms.entries()) {
            // 找到断开连接的玩家对应的角色
            let disconnectedRole = null;
            for (const [role, player] of room.playerMap.entries()) {
                if (player === ws) {
                    disconnectedRole = role;
                    break;
                }
            }
            
            if (disconnectedRole) {
                room.playerMap.delete(disconnectedRole);
                
                if (room.playerMap.size === 0) {
                    // 两个玩家都断开，删除房间
                    rooms.delete(roomId);
                    console.log(`房间 ${roomId} 已删除`);
                } else {
                    // 保留剩余玩家的角色不变
                    // 一个玩家断开，立即通知剩余玩家
                    broadcastToRoom(roomId, {
                        type: 'player_left',
                        message: '对手离开了游戏'
                    });
                    console.log(`玩家断开连接，房间 ${roomId} 通知剩余玩家`);
                }
                break;
            }
        }
    });
    
    ws.on('error', (error) => {
        console.error('连接错误:', error);
    });
});

// 处理消息
function handleMessage(ws, data) {
    switch (data.type) {
        case 'create_room':
            createRoom(ws, data);
            break;
        case 'join_room':
            joinRoom(ws, data);
            break;
        case 'game_action':
            broadcastGameAction(ws, data);
            break;
        case 'sync_state':
            broadcastSyncState(ws, data);
            break;
        case 'sync_response':
            handleSyncResponse(ws, data);
            break;
        // case 'action_request':
        //     broadcastActionRequest(ws, data);
        //     break;
        // case 'action_approved':
        //     broadcastActionResponse(ws, data);
        //     break;
        // case 'action_rejected':
        //     broadcastActionResponse(ws, data);
        //     break;
        default:
            console.log('未知消息类型:', data.type);
    }
}

// // 广播操作申请
// function broadcastActionRequest(ws, data) {
//     for (const [roomId, room] of rooms.entries()) {
//         if (room.players.includes(ws)) {
//             room.players.forEach(player => {
//                 if (player !== ws) {
//                     player.send(JSON.stringify({
//                         type: 'action_request',
//                         actionType: data.actionType
//                     }));
//                 }
//             });
//             break;
//         }
//     }
// }

// // 广播操作响应
// function broadcastActionResponse(ws, data) {
//     for (const [roomId, room] of rooms.entries()) {
//         if (room.players.includes(ws)) {
//             room.players.forEach(player => {
//                 if (player !== ws) {
//                     player.send(JSON.stringify({
//                         type: data.type,
//                         actionType: data.actionType
//                     }));
//                 }
//             });
//             break;
//         }
//     }
// }

// 创建房间
function createRoom(ws, data) {
    const roomId = generateRoomId();
    rooms.set(roomId, {
        playerMap: new Map(), // 存储角色和WebSocket的映射：{ 'first_player': ws, 'second_player': ws }
        host: ws,
        gameState: null,
        isGameStarted: false
    });
    // 房主加入房间，暂不分配角色，等待第二个玩家加入后随机分配
    rooms.get(roomId).playerMap.set('temp', ws);
    
    ws.send(JSON.stringify({
        type: 'room_created',
        roomId: roomId,
        message: '房间创建成功'
    }));
    
    console.log(`房间 ${roomId} 创建成功`);
}

// 加入房间
function joinRoom(ws, data) {
    const { roomId } = data;
    const room = rooms.get(roomId);
    
    if (room) {
        if (room.playerMap.size === 0) {
            // 房间已空，重新加入作为房主
            room.playerMap.set('first_player', ws);
            room.host = ws;
            
            ws.send(JSON.stringify({
                type: 'room_joined',
                roomId: roomId,
                message: '重新加入房间成功，等待对手'
            }));
            
            console.log(`玩家重新加入房间 ${roomId} 作为房主`);
        } else if (room.playerMap.size === 1) {
            // 房间有一个玩家，新玩家加入
            // 获取现有玩家和角色
            const existingEntries = Array.from(room.playerMap.entries());
            const [existingRole, existingPlayer] = existingEntries[0];
            
            if (existingRole === 'temp') {
                // 现有玩家是临时角色，随机分配角色
                const roles = ['first_player', 'second_player'];
                const firstRole = roles[Math.floor(Math.random() * 2)];
                const secondRole = roles.find(r => r !== firstRole);
                
                // 清空临时映射并分配角色
                room.playerMap.clear();
                room.playerMap.set(firstRole, existingPlayer);
                room.playerMap.set(secondRole, ws);
            } else {
                // 现有玩家已有角色，分配另一个角色
                const availableRole = existingRole === 'first_player' ? 'second_player' : 'first_player';
                room.playerMap.set(availableRole, ws);
            }
            
            // 通知新玩家加入成功
            ws.send(JSON.stringify({
                type: 'room_joined',
                roomId: roomId,
                message: '加入房间成功'
            }));
            
            if (!room.isGameStarted) {
                // 游戏尚未开始，开始游戏
                startGame(roomId);
            } else {
                // 游戏正在进行中，广播游戏继续，包含角色信息
                room.playerMap.forEach((player, role) => {
                    player.send(JSON.stringify({
                        type: 'game_continue',
                        message: '游戏继续',
                        role: role
                    }));
                });
                
                // 向现有玩家发送同步请求（发送给非新加入的玩家）
                room.playerMap.forEach((player, role) => {
                    if (player !== ws) {
                        player.send(JSON.stringify({
                            type: 'sync_request',
                            message: '请发送游戏状态'
                        }));
                    }
                });
            }
            
            console.log(`玩家加入房间 ${roomId}`);
        } else {
            ws.send(JSON.stringify({
                type: 'error',
                message: '房间已满'
            }));
        }
    } else {
        ws.send(JSON.stringify({
            type: 'error',
            message: '房间不存在'
        }));
    }
}

// 开始游戏
function startGame(roomId) {
    const room = rooms.get(roomId);
    if (room && room.playerMap.size === 2) {
        // 通知双方
        room.playerMap.get('first_player').send(JSON.stringify({
            type: 'game_start',
            role: 'first_player',
            message: '你是先手，请开始游戏'
        }));
        
        room.playerMap.get('second_player').send(JSON.stringify({
            type: 'game_start',
            role: 'second_player',
            message: '你是后手，等待先手操作'
        }));
        
        room.isGameStarted = true;
        console.log(`房间 ${roomId} 游戏开始`);
    }
}

// 广播游戏操作
function broadcastGameAction(ws, data) {
    for (const [roomId, room] of rooms.entries()) {
        // 找到包含当前WebSocket的房间
        for (const [role, player] of room.playerMap.entries()) {
            if (player === ws) {
                room.playerMap.forEach((player, role) => {
                    if (player !== ws) {
                        player.send(JSON.stringify({
                            type: 'game_action',
                            action: data.action,
                            shouldChangeTurn: data.shouldChangeTurn
                        }));
                    }
                });
                break;
            }
        }
    }
}

// 处理同步响应
function handleSyncResponse(ws, data) {
    for (const [roomId, room] of rooms.entries()) {
        // 找到包含当前WebSocket的房间
        for (const [role, player] of room.playerMap.entries()) {
            if (player === ws) {
                // 找到新加入的玩家（不是当前发送同步响应的玩家）
                room.playerMap.forEach((player, role) => {
                    if (player !== ws) {
                        // 向新玩家发送同步状态，只发送history
                        player.send(JSON.stringify({
                            type: 'sync_state',
                            history: data.history,
                            isMyTurn: data.isMyTurn
                        }));
                        console.log(`向新玩家发送同步状态，房间 ${roomId}`);
                    }
                });
                break;
            }
        }
    }
}

// 广播状态同步
function broadcastSyncState(ws, data) {
    for (const [roomId, room] of rooms.entries()) {
        // 找到包含当前WebSocket的房间
        for (const [role, player] of room.playerMap.entries()) {
            if (player === ws) {
                room.gameState = data.state;
                room.playerMap.forEach((player, role) => {
                    if (player !== ws) {
                        player.send(JSON.stringify({
                            type: 'sync_state',
                            state: data.state,
                            isMyTurn: data.isMyTurn
                        }));
                    }
                });
                break;
            }
        }
    }
}

// 广播消息到房间
function broadcastToRoom(roomId, message) {
    const room = rooms.get(roomId);
    if (room) {
        room.playerMap.forEach(player => {
            player.send(JSON.stringify(message));
        });
    }
}

// 生成房间ID
function generateRoomId() {
    return Math.random().toString(36).substring(2, 8).toUpperCase();
}

// 启动服务器
const PORT = 8080;
server.listen(PORT, '0.0.0.0', () => {
    console.log(`WebSocket服务器启动在端口 ${PORT}`);
    console.log(`本地访问地址: ws://localhost:${PORT}`);
    console.log(`局域网访问地址: ws://<本机IP>:${PORT}`);
    console.log('请在客户端输入服务器的局域网IP地址进行连接');
});

console.log('WebSocket服务器已启动');
