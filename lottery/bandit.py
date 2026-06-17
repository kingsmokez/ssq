# -*- coding: utf-8 -*-
"""Thompson Sampling 动态策略分配 — 用贝叶斯多臂老虎机自适应选择最优策略。

核心思路：
  有 5 种选号策略（热号/冷号/趋势/合规/幸运）+ ML 评分 + 随机基线。
  每次开奖后，根据各策略的历史命中表现，用 Thompson Sampling
  动态决定下一期应该用哪个策略（或如何加权混合）。

Thompson Sampling 原理：
  每个策略的命中率 θ_k ~ Beta(α_k + hits_k, β_k + misses_k)
  每轮从每个策略的后验分布中采样一个 θ_k，
  选择 θ_k 最大的策略（或用 θ_k 作为混合权重）。

这比固定策略 + 递增多样性惩罚更灵活——它会自动：
  - 给近期表现好的策略更多权重
  - 给长期表现差的策略更少权重
  - 在探索（试新策略）和利用（用已知好策略）之间自动平衡
"""

import random as rnd
import math
from typing import Dict, List, Tuple
from collections import defaultdict


class ThompsonBandit:
    """Thompson Sampling 多臂老虎机用于策略选择。"""

    def __init__(self, strategy_names: List[str], prior_a: float = 1.0,
                 prior_b: float = 1.0):
        """初始化。

        strategy_names: 策略名列表
        prior_a, prior_b: Beta 先验参数（1,1=均匀先验）
        """
        self.strategies = strategy_names
        self.prior_a = prior_a
        self.prior_b = prior_b
        # 每个策略的 (hits, misses) 计数
        self.counts = {s: {"hits": prior_a, "misses": prior_b}
                       for s in strategy_names}
        # 最近 N 轮的滑动窗口计数（更关注近期表现）
        self.recent_window = 20
        self.recent_counts = {s: {"hits": 0, "misses": 0, "total": 0}
                              for s in strategy_names}
        self.history: List[Dict] = []  # 历史记录

    def update(self, strategy: str, hit: bool, n_picks: int = 1):
        """更新某策略的命中统计。

        strategy: 策略名
        hit: 是否命中
        n_picks: 本轮选了几个号（用于计算命中比例）
        """
        if strategy not in self.counts:
            return

        if hit:
            self.counts[strategy]["hits"] += 1
            self.recent_counts[strategy]["hits"] += 1
        else:
            self.counts[strategy]["misses"] += 1
            self.recent_counts[strategy]["misses"] += 1

        self.recent_counts[strategy]["total"] += 1

        # 滑动窗口：如果某个策略的近期总计超过窗口大小，衰减旧数据
        for s in self.recent_counts:
            if self.recent_counts[s]["total"] > self.recent_window:
                # 按比例衰减
                scale = self.recent_window / self.recent_counts[s]["total"]
                self.recent_counts[s]["hits"] = max(0, int(self.recent_counts[s]["hits"] * scale))
                self.recent_counts[s]["misses"] = max(0, int(self.recent_counts[s]["misses"] * scale))
                self.recent_counts[s]["total"] = self.recent_counts[s]["hits"] + self.recent_counts[s]["misses"]

        self.history.append({
            "strategy": strategy, "hit": hit, "n_picks": n_picks,
        })

    def sample(self, use_recent: bool = True) -> str:
        """Thompson Sampling：从每个策略的后验分布采样，返回最优策略名。

        use_recent: True=用近期窗口（自适应快），False=用全历史（稳定但慢）
        """
        best_strategy = self.strategies[0]
        best_theta = -1

        for s in self.strategies:
            if use_recent:
                a = self.prior_a + self.recent_counts[s]["hits"]
                b = self.prior_b + max(0, self.recent_counts[s]["misses"])
            else:
                a = self.counts[s]["hits"]
                b = self.counts[s]["misses"]

            # Beta(a, b) 采样
            theta = rnd.betavariate(max(1, a), max(1, b))
            if theta > best_theta:
                best_theta = theta
                best_strategy = s

        return best_strategy

    def get_weights(self, use_recent: bool = True) -> Dict[str, float]:
        """获取各策略的混合权重（用后验均值）。

        返回 {strategy: weight}，所有权重和为 1。
        """
        weights = {}
        total = 0.0

        for s in self.strategies:
            if use_recent:
                a = self.prior_a + self.recent_counts[s]["hits"]
                b = self.prior_b + max(0, self.recent_counts[s]["misses"])
            else:
                a = self.counts[s]["hits"]
                b = self.counts[s]["misses"]

            # Beta 分布的均值 = a / (a + b)
            weight = a / (a + b) if (a + b) > 0 else 0.5
            weights[s] = weight
            total += weight

        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        return weights

    def get_stats(self) -> List[dict]:
        """获取各策略的统计信息。"""
        stats = []
        for s in self.strategies:
            a_all = self.counts[s]["hits"]
            b_all = self.counts[s]["misses"]
            a_rc = self.recent_counts[s]["hits"]
            b_rc = self.recent_counts[s]["misses"]
            rate_all = a_all / (a_all + b_all) if (a_all + b_all) > 0 else 0
            rate_rc = a_rc / (a_rc + b_rc) if (a_rc + b_rc) > 0 else 0
            stats.append({
                "strategy": s,
                "hit_rate_all": round(rate_all, 4),
                "hit_rate_recent": round(rate_rc, 4),
                "total_trials": a_all + b_all - self.prior_a - self.prior_b,
                "recent_trials": a_rc + b_rc,
            })
        stats.sort(key=lambda x: -x["hit_rate_recent"])
        return stats


def bandit_backtest(game: str, n_test: int = 60) -> dict:
    """用 Thompson Sampling 动态选策略做回测。

    每轮：
      1. 用 bandit 选一个策略
      2. 用该策略生成推荐
      3. 对比实际开奖，更新 bandit

    对比：固定策略 vs bandit 动态选择
    """
    from . import models
    from .analyzer import analyze_from_records
    from .recommender import recommend
    from .backtester import _build_window
    from config import GAMES

    cfg = GAMES[game]
    is_lotto = cfg["type"] == "lotto"

    all_records = models.fetch_draws(game, limit=3000, order_desc=True)
    if len(all_records) < n_test + 50:
        n_test = max(5, len(all_records) - 50)
    if len(all_records) < 30:
        return {"error": "数据不足"}

    strategy_names = ["均衡热号", "冷号回补", "趋势加速", "组合合规", "幸运组合"]
    bandit = ThompsonBandit(strategy_names)

    bandit_hits = bandit_picks = 0
    best_fixed_hits = best_fixed_picks = 0
    n_points = 0
    strategy_used = defaultdict(int)

    step = max(1, n_test // 80)
    max_i = min(n_test, len(all_records) - 30)

    for i in range(1, max_i, step):
        test_rec = all_records[i]
        train_records = all_records[i + 1: i + 800]
        if len(train_records) < 30:
            continue
        train_window = _build_window(train_records, 730)
        if len(train_window) < 20:
            train_window = train_records

        gs = analyze_from_records(game, train_window)
        rec = recommend(game, gs)

        # ---- Bandit 选择策略 ----
        chosen = bandit.sample(use_recent=True)
        strategy_used[chosen] += 1

        # 找到对应组
        chosen_group = None
        for grp in rec["single"]:
            if grp["label"] == chosen:
                chosen_group = grp
                break

        if is_lotto and chosen_group:
            actual_front = set(test_rec["front"])
            front_picks = chosen_group["picks"]["front"]
            bandit_hits += len(set(front_picks) & actual_front)
            bandit_picks += len(front_picks)
            # 更新 bandit
            hit_count = len(set(front_picks) & actual_front)
            bandit.update(chosen, hit_count > 0, n_picks=len(front_picks))
        elif chosen_group:
            actual = list(test_rec["front"])
            digits = chosen_group["picks"]["digits"]
            pos_hits = sum(1 for j in range(min(len(digits), len(actual)))
                          if digits[j] == actual[j])
            bandit_hits += pos_hits
            bandit_picks += len(digits)
            bandit.update(chosen, pos_hits > 0, n_picks=len(digits))

        # ---- 最佳固定策略（取本轮所有组中最好的） ----
        best_in_round = 0
        for grp in rec["single"]:
            if is_lotto and "front" in grp.get("picks", {}):
                hits = len(set(grp["picks"]["front"]) & set(test_rec["front"]))
                if hits > best_in_round:
                    best_in_round = hits
        if not is_lotto:
            for grp in rec["single"]:
                digits = grp.get("picks", {}).get("digits", [])
                hits = sum(1 for j in range(min(len(digits), len(test_rec["front"])))
                          if digits[j] == test_rec["front"][j])
                if hits > best_in_round:
                    best_in_round = hits

        if is_lotto:
            best_fixed_hits += best_in_round
            best_fixed_picks += 5  # 前区5个
        else:
            best_fixed_hits += best_in_round
            best_fixed_picks += len(test_rec["front"])

        n_points += 1

    bandit_rate = bandit_hits / bandit_picks if bandit_picks else 0
    fixed_rate = best_fixed_hits / best_fixed_picks if best_fixed_picks else 0
    lift = (bandit_rate - fixed_rate) / fixed_rate if fixed_rate > 0 else 0

    return {
        "bandit_hit_rate": round(bandit_rate, 4),
        "best_fixed_hit_rate": round(fixed_rate, 4),
        "lift": round(lift, 4),
        "bandit_hits": bandit_hits,
        "bandit_picks": bandit_picks,
        "n_points": n_points,
        "strategy_usage": dict(strategy_used),
        "bandit_stats": bandit.get_stats(),
    }
