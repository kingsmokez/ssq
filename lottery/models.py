# -*- coding: utf-8 -*-
"""数据库模型与读写 helper（SQLite）。

表结构：
  draws             历史开奖（game, issue, draw_date, numbers_json, front_json, back_json）
  recommendations   推荐记录（game, created_at, mode, picks_json, reason, scores_json）
  refresh_log       刷新日志（created_at, game, new_count, status, message）
"""

import json
import os
import sqlite3
from datetime import datetime

from config import DB_PATH, DATA_DIR

DDL = """
CREATE TABLE IF NOT EXISTS draws (
    game        TEXT NOT NULL,
    issue       TEXT NOT NULL,
    draw_date   TEXT,                 -- YYYY-MM-DD
    numbers     TEXT,                 -- 所有号码 JSON（lotto: front+back 平铺; digit: 各位）
    front       TEXT,                 -- 前区/主号码 JSON
    back        TEXT,                 -- 后区 JSON（digit 玩法为 []）
    created_at  TEXT NOT NULL,
    PRIMARY KEY (game, issue)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    game        TEXT NOT NULL,
    group_index INTEGER NOT NULL,     -- 单选第几组 (1-5) 或复式档位标记
    mode        TEXT NOT NULL,        -- single / small / medium
    label       TEXT,                 -- 组别名（均衡热号/冷号回补...）
    picks       TEXT NOT NULL,        -- JSON
    reason      TEXT,                 -- 推荐理由
    scores      TEXT,                 -- JSON 快照
    bets        INTEGER DEFAULT 1,    -- 注数
    cost        INTEGER DEFAULT 2,    -- 投注金额（元）
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS refresh_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    game        TEXT,
    new_count   INTEGER DEFAULT 0,
    status      TEXT,                 -- ok / error
    message     TEXT
);

CREATE INDEX IF NOT EXISTS idx_draws_game_date ON draws(game, draw_date);
CREATE INDEX IF NOT EXISTS idx_reco_game_time  ON recommendations(game, created_at);
"""


def get_conn() -> sqlite3.Connection:
    """返回一个 Row 工厂的连接。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """建表（幂等）+ 增量迁移。"""
    conn = get_conn()
    try:
        conn.executescript(DDL)
        # 增量迁移：为旧表添加 bets/cost 列
        cols = [r[1] for r in conn.execute("PRAGMA table_info(recommendations)").fetchall()]
        if "bets" not in cols:
            conn.execute("ALTER TABLE recommendations ADD COLUMN bets INTEGER DEFAULT 1")
        if "cost" not in cols:
            conn.execute("ALTER TABLE recommendations ADD COLUMN cost INTEGER DEFAULT 2")
        conn.commit()
    finally:
        conn.close()


# ---------------- draws 读写 ----------------

def upsert_draw(game: str, issue: str, draw_date, numbers, front, back) -> bool:
    """插入或忽略一条开奖记录。返回是否新增。"""
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO draws "
            "(game, issue, draw_date, numbers, front, back, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                game, str(issue),
                draw_date or "",
                json.dumps(numbers, ensure_ascii=False),
                json.dumps(front, ensure_ascii=False),
                json.dumps(back, ensure_ascii=False),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def _decode_row(row: dict) -> dict:
    """将一行的 JSON 字符串字段解码为 list（numbers/front/back）。"""
    if row is None:
        return row
    for k in ("numbers", "front", "back"):
        v = row.get(k)
        if isinstance(v, str) and v:
            try:
                row[k] = json.loads(v)
            except (ValueError, TypeError):
                row[k] = []
        elif v is None:
            row[k] = []
    return row


def fetch_draws(game: str, limit: int = 2000, order_desc: bool = True):
    """返回某玩法开奖记录（按期号/日期排序），最新在前（默认）。"""
    conn = get_conn()
    try:
        direction = "DESC" if order_desc else "ASC"
        rows = conn.execute(
            f"SELECT * FROM draws WHERE game=? ORDER BY draw_date {direction}, issue {direction} LIMIT ?",
            (game, limit),
        ).fetchall()
        return [_decode_row(dict(r)) for r in rows]
    finally:
        conn.close()


def fetch_draws_since(game: str, since_date: str, limit: int = 5000):
    """返回某玩法自 since_date(含) 起的记录，最新在前。"""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM draws WHERE game=? AND draw_date>=? "
            "ORDER BY draw_date DESC, issue DESC LIMIT ?",
            (game, since_date, limit),
        ).fetchall()
        return [_decode_row(dict(r)) for r in rows]
    finally:
        conn.close()


def latest_draw(game: str):
    """返回某玩法最新一期（或 None）。"""
    rows = fetch_draws(game, limit=1, order_desc=True)
    return rows[0] if rows else None


def count_draws(game: str) -> int:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM draws WHERE game=?", (game,)
        ).fetchone()
        return row["c"] if row else 0
    finally:
        conn.close()


def get_all_games_latest():
    """返回 {game: latest_draw_row} 用于 dashboard。"""
    result = {}
    for game in ("dlt", "pl3", "pl5"):
        result[game] = latest_draw(game)
    return result


# ---------------- recommendations 读写 ----------------

def save_recommendation(game, group_index, mode, label, picks, reason, scores,
                        created_at=None, bets=1, cost=2):
    conn = get_conn()
    try:
        ts = created_at or datetime.now().isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO recommendations "
            "(game, group_index, mode, label, picks, reason, scores, bets, cost, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                game, group_index, mode, label,
                json.dumps(picks, ensure_ascii=False),
                reason,
                json.dumps(scores, ensure_ascii=False),
                bets, cost,
                ts,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def latest_recommendations(game: str, mode: str = "single"):
    """返回某玩法某档位最新一批推荐（按 created_at 取最近一批，按 group_index 排序）。"""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT created_at FROM recommendations WHERE game=? AND mode=? "
            "ORDER BY created_at DESC LIMIT 1",
            (game, mode),
        ).fetchone()
        if not row:
            return []
        rows = conn.execute(
            "SELECT * FROM recommendations WHERE game=? AND mode=? AND created_at=? "
            "ORDER BY group_index ASC",
            (game, mode, row["created_at"]),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["picks"] = json.loads(d["picks"])
            d["scores"] = json.loads(d["scores"]) if d["scores"] else {}
            out.append(d)
        return out
    finally:
        conn.close()


def all_recommendation_batches(game: str, limit: int = 30):
    """返回某玩法历史推荐批次（用于回测），按时间倒序。返回 list of (created_at, mode, picks list)。"""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM recommendations WHERE game=? "
            "ORDER BY created_at DESC LIMIT ?",
            (game, limit * 10),
        ).fetchall()
        from collections import defaultdict
        batches = defaultdict(list)
        for r in rows:
            d = dict(r)
            d["picks"] = json.loads(d["picks"])
            batches[(d["created_at"], d["mode"])].append(d)
        return batches
    finally:
        conn.close()


# ---------------- refresh_log ----------------

def log_refresh(game, new_count, status, message):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO refresh_log (created_at, game, new_count, status, message) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), game, new_count, status, message),
        )
        conn.commit()
    finally:
        conn.close()


def recent_refresh_logs(limit: int = 20):
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM refresh_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"数据库已初始化: {DB_PATH}")
    for g in ("dlt", "pl3", "pl5"):
        print(f"  {g}: {count_draws(g)} 条记录")
