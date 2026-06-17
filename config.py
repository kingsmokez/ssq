# -*- coding: utf-8 -*-
"""
体彩推荐分析系统 — 全局配置

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
# name: (中文名, 前区/主号码池范围, 前区选几个, 后区范围, 后区选几个, 是否每日开奖)
#   - 数字型玩法 (pl3/pl5) 把"每位 0-9"建模为候选池 [0..9]，每位独立；
#     为统一数据结构，pl3/pl5 的 front = 各位数字列表，back 为空列表。
GAMES = {
    "dlt": {
        "name": "超级大乐透",
        "type": "lotto",                 # lotto: 选号池; digit: 每位数字
        "front_pool": list(range(1, 36)),   # 前区 1-35
        "front_pick": 5,
        "back_pool": list(range(1, 13)),    # 后区 1-12
        "back_pick": 2,
        "draw_days": [1, 3, 6],             # 周一/三/六 (0=周一)
    },
    "pl3": {
        "name": "排列3",
        "type": "digit",
        "positions": 3,                     # 3 位
        "digit_pool": list(range(0, 10)),   # 每位 0-9
        "draw_days": list(range(0, 7)),     # 每日
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

# 备选数据源（当主源不可用时自动降级）
# kaijiang.500.com 是 500.com 的另一个开奖数据接口
FALLBACK_PATHS = {
    "dlt": "/dlt/history/newinc/history.php",     # 与主源相同路径，不同域名
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
REQUEST_TIMEOUT = 25          # 秒
REQUEST_SLEEP = 1.0           # 每次请求间隔（礼貌限速）
REQUEST_RETRIES = 2           # 失败重试次数
# 首次抓取目标期数（近 2 年：大乐透约 300 期，排列约 1400 期，此处统一取较大值）
INITIAL_FETCH_LIMIT = 2000

# ---------- 统计窗口 ----------
STATS_WINDOW_DAYS = 730       # 近 2 年
MOMENTUM_RECENT_N = 30        # 近 30 期用于趋势
COLD_HOT_RECENT_N = 50        # 近 50 期用于冷热标签

# ---------- 评分权重（可调） ----------
WEIGHTS = {
    "freq": 0.45,             # 频率
    "omission": 0.30,         # 遗漏比
    "momentum": 0.25,         # 近期趋势
}

# ---------- 复式档位 ----------
COMBO_MODES = {
    # lotto: (前区选几个, 后区选几个)
    "lotto": {
        "single": (5, 2),
        "small":  (6, 3),     # 12 注
        "medium": (8, 4),     # 112 注
    },
    # digit: 每位给几个候选号
    "digit": {
        "single": 1,
        "small":  2,
        "medium": 3,
    },
}

# ---------- 投注单价（元/注） ----------
BET_UNIT_PRICE = {
    "dlt": 2,    # 大乐透 2 元/注
    "pl3": 2,    # 排列3  2 元/注
    "pl5": 2,    # 排列5  2 元/注
}

# ---------- 组合约束阈值 ----------
BIG_SMALL_THRESHOLD = 18        # 大乐透前区大小号分界（>=18为大，<18为小）
DIGIT_COMBO_TOP_K = 3           # 排列玩法组合合规组每位候选数

# ---------- 资金管理（理性购彩参考阈值）----------
# 基于常识"娱乐性支出不应超过可支配收入1%"，非投资建议。
# 前端 dashboard 与 ev_calc.budget_warning 共用。
MONTHLY_BUDGET_LIMIT = 200      # 建议月投入上限（元）
CHASE_LOSS_LIMIT = 5            # 连续追号期数上限（止损）

# ---------- 免责声明 ----------
DISCLAIMER = (
    "⚠️ 免责声明：彩票每次开奖均为独立随机事件，任何算法都无法真正预测中奖号码。"
    "本系统基于历史统计数据给出参考推荐，仅供娱乐与研究，不构成任何投注建议。请理性购彩，量力而行。"
)
