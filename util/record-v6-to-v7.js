#!/usr/bin/env node
/**
 * 棋局记录格式转换工具（v6 <-> v7）
 *
 * 版本说明:
 *   v6 - 历史记录为对象格式，如 {"opt":"res","x":33,"y":21}
 *   v7 - 历史记录为紧凑数组格式，如 ["r",33,21]（编码规则见 index.html 的
 *        HISTORY_OPT_TO_CODE，本文件需与之一致）
 *
 * 用法:
 *   node util/record-v6-to-v7.js to7 [文件或目录]   # v6 -> v7
 *   node util/record-v6-to-v7.js to6 [文件或目录]   # v7 -> v6
 *   未指定路径时默认转换 records/ 目录下所有 .json 文件
 */
'use strict';

const fs = require('fs');
const path = require('path');

// 与 index.html 中的 HISTORY_OPT_TO_CODE / HISTORY_PLAYER_CODE 保持一致
const OPT_TO_CODE = {
    init: 'i', res: 'r', eptRes: 'e',
    mainX: 'mx', mainO: 'mo', subX: 'sx', subO: 'so',
    layX: 'lx', layO: 'lo',
    'x-place': 'px', 'o-place': 'po', 'x-eat': 'ex', 'o-eat': 'eo',
    choose: 'c', resources: 's', select: 'sl'
};
const PLAYER_CODE = { cross: 'x', circle: 'o' };
const PLAYER_NAME = { x: 'cross', o: 'circle' };

// 内部 history 对象 -> 紧凑数组（v7）
function encodeHistory(history) {
    return history.map(a => {
        const code = OPT_TO_CODE[a.opt];
        switch (a.opt) {
            case 'init': return ['i'];
            case 'res': case 'eptRes': case 'mainX': case 'mainO': case 'subX': case 'subO': case 'layX': case 'layO':
                return [code, a.x, a.y];
            case 'x-place': case 'o-place': case 'x-eat': case 'o-eat':
                return [code, a.x, a.y, a.fromX, a.fromY];
            case 'choose':
                return ['c', PLAYER_CODE[a.player] || a.player, a.x, a.y];
            case 'resources':
                return ['s', a.resources.map(r => [r.x, r.y])];
            case 'select':
                return ['sl', a.x, a.y];
            default:
                return a; // 未知操作保持原样
        }
    });
}

// 紧凑数组 -> 内部 history 对象（非数组条目为旧 v6，直接透传）
function decodeHistory(history) {
    return history.map(a => {
        if (!Array.isArray(a)) return a;
        const [code, b, c, d, e] = a;
        switch (code) {
            case 'i': return { opt: 'init' };
            case 'r': return { opt: 'res', x: b, y: c };
            case 'e': return { opt: 'eptRes', x: b, y: c };
            case 'mx': return { opt: 'mainX', x: b, y: c };
            case 'mo': return { opt: 'mainO', x: b, y: c };
            case 'sx': return { opt: 'subX', x: b, y: c };
            case 'so': return { opt: 'subO', x: b, y: c };
            case 'lx': return { opt: 'layX', x: b, y: c };
            case 'lo': return { opt: 'layO', x: b, y: c };
            case 'px': return { opt: 'x-place', fromX: d, fromY: e, x: b, y: c };
            case 'po': return { opt: 'o-place', fromX: d, fromY: e, x: b, y: c };
            case 'ex': return { opt: 'x-eat', fromX: d, fromY: e, x: b, y: c };
            case 'eo': return { opt: 'o-eat', fromX: d, fromY: e, x: b, y: c };
            case 'c': return { opt: 'choose', player: PLAYER_NAME[b] || b, x: c, y: d };
            case 's': return { opt: 'resources', resources: b.map(([x, y]) => ({ x, y })) };
            case 'sl': return { opt: 'select', x: b, y: c };
            default: return a;
        }
    });
}

// 依据 meta.version 判断格式；缺失时按 history 首条目是否为数组推断
function detectVersion(data) {
    if (data.meta && data.meta.version) return data.meta.version;
    const h = data.history;
    if (Array.isArray(h) && h.length > 0 && Array.isArray(h[0])) return '7';
    return '6';
}

function convertFile(filePath, targetVer) {
    const raw = fs.readFileSync(filePath, 'utf8');
    const data = JSON.parse(raw);
    const fromVer = detectVersion(data);
    if (fromVer === targetVer) {
        console.log(`[跳过] ${path.basename(filePath)} 已是 v${targetVer}`);
        return { skipped: true };
    }
    const oldSize = Buffer.byteLength(raw, 'utf8');
    if (targetVer === '7') {
        data.history = encodeHistory(data.history);
    } else {
        data.history = decodeHistory(data.history);
    }
    data.meta = data.meta || {};
    data.meta.version = targetVer;
    const out = JSON.stringify(data);
    fs.writeFileSync(filePath, out, 'utf8');
    const newSize = Buffer.byteLength(out, 'utf8');
    console.log(`[完成] ${path.basename(filePath)} v${fromVer} -> v${targetVer} (${oldSize}B -> ${newSize}B, ${Math.round((1 - newSize / oldSize) * 100)}%)`);
    return { skipped: false, oldSize, newSize };
}

function collectJsonFiles(target) {
    const full = path.resolve(target);
    if (fs.statSync(full).isDirectory()) {
        return fs.readdirSync(full).filter(f => f.toLowerCase().endsWith('.json')).map(f => path.join(full, f));
    }
    return [full];
}

function printUsage() {
    console.log('用法: node util/record-v6-to-v7.js <to7|to6> [文件或目录]');
    console.log('  未指定路径时默认转换 records/ 目录下所有 .json 文件');
}

function main(args) {
    const direction = (args[0] || '').toLowerCase();
    if (direction !== 'to7' && direction !== 'to6') {
        printUsage();
        process.exit(1);
    }
    const target = args[1] || path.join(__dirname, '..', 'records');
    const targetVer = direction === 'to7' ? '7' : '6';

    let files;
    try {
        files = collectJsonFiles(target);
    } catch (e) {
        console.error('路径不存在或无法访问:', target);
        process.exit(1);
    }
    if (files.length === 0) {
        console.log('未找到 .json 文件:', target);
        return;
    }

    console.log(`开始转换 ${files.length} 个文件 -> v${targetVer} ...`);
    let done = 0, skipped = 0, oldTotal = 0, newTotal = 0;
    for (const f of files) {
        try {
            const r = convertFile(f, targetVer);
            if (r.skipped) { skipped++; } else { done++; oldTotal += r.oldSize; newTotal += r.newSize; }
        } catch (e) {
            console.error(`[失败] ${path.basename(f)}: ${e.message}`);
        }
    }
    console.log(`--- 完成: 转换 ${done} 个, 跳过 ${skipped} 个` + (done ? `, 总体积 ${oldTotal}B -> ${newTotal}B (${Math.round((1 - newTotal / oldTotal) * 100)}%)` : ''));
}

main(process.argv.slice(2));
