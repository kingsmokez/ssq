# -*- coding: utf-8 -*-
"""体彩推荐分析系统 — 全局配置

⚠️ 免责声明：彩票每次开奖均为独立随机事件，任何算法都无法预测中奖号码。
   本系统基于历史统计做"最符合历史规律"的参考推荐，仅供娱乐与研究，不构成投注建议。
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 路径 ----------
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "lottery.db")

# ---------- Flask ----------
SECRET_KEY = "bocai-change-this-secret-key-to-a-random-string"
LOGIN_PASSWORD = "admin123"          # 登录密码，请自行修改
HOST = "127.0.0.1"                   # 仅本机访问
PORT = 5000

# ---------- 玩法规则 ----------
GAMES = {
    "dlt": {
        "name": "超级大乐透",
        "type": "lotto",
        "front_pool": list(range(1, 36)),
        "front_pick": 5,
        "back_pool": list(range(1, 13)),
        "back_pick": 2,
        "draw_days": [1, 3, 6],
    },
    "pl3": {
        "name": "排列3",
        "type": "digit",
        "positions": 3,
        "digit_pool": list(range(0, 10)),
        "draw_days": list(range(0, 7)),
    },
    "pl5": {
        "name": "排列5",
        "type": "digit",
        "positions": 5,
        "digit_pool": list(range(0, 10)),
        "draw_days": list(range(0, 7)),
    },
}

GAME_ORDER = ["dlt", "pl3", "pl5"]

# ---------- 数据源 (500.com datachart) ----------
SOURCE_URLS = {
    "dlt": "https://datachart.500.com/dlt/history/history.shtml",
    "pl3": "https://datachart.500.com/pls/history/history.shtml",
    "pl5": "https://datachart.500.com/plw/history/history.shtml",
}
SOURCE_NAME = "500彩票网 datachart"

FALLBACK_PATHS = {
    "dlt": "/dlt/history/newinc/history.php",
    "pl3": "/pls/history/inc/history.php",
    "pl5": "/plw/history/inc/history.php",
}
FALLBACK_BASE = "https://kaijiang.500.com"
FALLBACK_NAME = "500.com 开奖网（备选）"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://datachart.500.com/",
}
REQUEST_TIMEOUT = 25
REQUEST_SLEEP = 1.0
REQUEST_RETRIES = 2
INITIAL_FETCH_LIMIT = 2000

# ---------- 统计窗口 ----------
STATS_WINDOW_DAYS = 730
MOMENTUM_RECENT_N = 30
COLD_HOT_RECENT_N = 50
EWMA_HALF_LIFE = 25

# ---------- 评分权重（传统评分，不含条件概率维度） ----------
# 条件概率由 _fused_score 独立计算，不再嵌入 _score_map
WEIGHTS = {
    "freq": 0.45,             # 频率（EWMA + 贝叶斯平滑）
    "omission": 0.30,         # 遗漏比
    "momentum": 0.25,         # 近期趋势
}

# ---------- 多信号融合权重 ----------
# 生产推荐使用多信号融合，而非单一 _score_map
SIGNAL_WEIGHTS = {
    "traditional": 0.30,      # 传统评分（频率+遗漏+趋势）
    "multi_window": 0.25,     # 多窗口特征融合
    "conditional": 0.25,      # 条件概率信号
    "arima": 0.20,            # AR(1) 时间序列预测
}

# ---------- 多窗口配置 ----------
WINDOW_CONFIGS = [
    {"name": "short",   "days": 90,   "momentum_n": 10, "cold_n": 20,  "weight": 0.20},
    {"name": "medium",  "days": 365,  "momentum_n": 25, "cold_n": 40,  "weight": 0.35},
    {"name": "long",    "days": 730,  "momentum_n": 30, "cold_n": 50,  "weight": 0.30},
    {"name": "very_long", "days": 1095, "momentum_n": 50, "cold_n": 80, "weight": 0.15},
]

# ---------- 反冷号策略阈值 ----------
ANTI_COLD_MAX_OMISSION_RATIO = 1.5   # 遗漏比超过此值的号被排除（活跃号筛选）
ANTI_COLD_MIN_RECENT_APPEAR = 1      # 近N期至少出现1次才算活跃号

# ---------- 软多样性惩罚参数 ----------
DIVERSITY_BASE_PENALTY = 0.50        # 基础惩罚（已选号降权50%，比v3的30%更强）
DIVERSITY_REPEAT_PENALTY = 0.75      # 重复选号惩罚（已选2次降权75%）
DIVERSITY_HARD_EXCLUDE_THRESHOLD = 3  # 已选超过此次则硬排除

# ---------- 位置感知评分参数 ----------
POSITION_AWARE_TOP_N = 8             # 每个位置取 Top-N 候选号
POSITION_AWARE_WEIGHT = 0.40         # 位置评分融合权重（0=纯整体，1=纯位置）

# ---------- 集成投票参数 ----------
ENSEMBLE_VOTE_WEIGHT_DECAY = 0.85    # 按组序号衰减权重（组1=1.0, 组2=0.85, ...）
ENSEMBLE_MIN_VOTES = 2               # 最少获得2票才可入选集成组

# ---------- 复式档位 ----------
COMBO_MODES = {
    "lotto": {
        "single": (5, 2),
        "small":  (6, 3),
        "medium": (8, 4),
    },
    "digit": {
        "single": 1,
        "small":  2,
        "medium": 3,
    },
}

# ---------- 投注单价（元/注） ----------
BET_UNIT_PRICE = {
    "dlt": 2,
    "pl3": 2,
    "pl5": 2,
}

# ---------- 组合约束阈值 ----------
BIG_SMALL_THRESHOLD = 18
DIGIT_COMBO_TOP_K = 3

# ---------- 资金管理 ----------
MONTHLY_BUDGET_LIMIT = 200
CHASE_LOSS_LIMIT = 5

# ---------- 免责声明 ----------
DISCLAIMER = (
    "⚠️ 免责声明：彩票每次开奖均为独立随机事件，任何算法都无法真正预测中奖号码。"
    "本系统基于历史统计数据给出参考推荐，仅供娱乐与研究，不构成任何投注建议。请理性购彩，量力而行。"
)
