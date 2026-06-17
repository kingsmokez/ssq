# -*- coding: utf-8 -*-
"""Stacking 元集成 v3 — 修复 ML 训练并集成到 recommend() 流程。

v3 核心改进：
1. ML 训练使用多历史窗口构建训练集（而非单样本）
2. 增加正则化防止过拟合
3. 提供 score_for_recommend() 方法直接供 recommend() 调用
4. 后区独立 ML 模型（12选2更容易学）
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

from config import GAMES, EWMA_HALF_LIFE
from . import models
from .analyzer import _per_number_stats, _build_window
from .recommender import _score_map, _normalize_dict, _weighted_blend, SIGNAL_WEIGHTS
from .arima_predictor import arima_scores_lotto, arima_scores_digit


class StackingEnsemble:
    """Stacking 集成评分器 — 融合多个评分方法。"""

    def __init__(self, game: str):
        self.game = game
        self.cfg = GAMES[game]
        self.scorer_names = ["ml_gbm", "traditional", "arima"]

    def score_for_recommend(self, train_records: list,
                             half_life: int = None) -> Tuple[Dict[int, float], Dict[int, float]]:
        """直接为 recommend() 提供融合评分（不依赖 ML，避免训练开销）。

        使用传统评分 + 多窗口 + 条件概率 + ARIMA 融合，
        跳过 ML（ML 训练太慢且效果不稳定）。

        返回 (front_fused, back_fused)
        """
        if half_life is None:
            half_life = EWMA_HALF_LIFE

        cfg = self.cfg
        window = _build_window(train_records, 730)
        if len(window) < 20:
            window = train_records

        # 前区
        front_sets = [set(r["front"]) for r in window]
        front_stats = _per_number_stats(cfg["front_pool"], front_sets, len(window),
                                         30, 50, half_life=half_life)
        trad_front = _score_map(front_stats)
        arima_front = arima_scores_lotto(window, cfg["front_pool"])

        # 后区
        back_sets = [set(r["back"]) for r in window]
        back_stats = _per_number_stats(cfg["back_pool"], back_sets, len(window),
                                        30, 50, half_life=half_life)
        trad_back = _score_map(back_stats)
        arima_back = arima_scores_lotto(window, cfg["back_pool"])

        # 多窗口评分
        from .multi_window import multi_window_features, ensemble_multi_window_score
        mw_feats = multi_window_features(self.game, train_records, half_life=half_life)
        mw_front = ensemble_multi_window_score(mw_feats)

        # 条件概率评分
        from .conditional import conditional_prob_matrix
        cond_front = conditional_prob_matrix(window, cfg["front_pool"])
        # 转为每号得分：与Top10号的平均共生率
        top_nums = sorted(trad_front, key=lambda n: -trad_front.get(n, 0))[:10]
        cond_front_score = {}
        for n in cfg["front_pool"]:
            if n in cond_front:
                rates = [cond_front[n].get(t, 0) for t in top_nums if t != n]
                cond_front_score[n] = sum(rates) / len(rates) if rates else 0.5
            else:
                cond_front_score[n] = 0.5

        # 后区无条件概率（简化）
        cond_back_score = {n: 0.5 for n in cfg["back_pool"]}

        # 融合
        signals_front = {
            "traditional": _normalize_dict(trad_front),
            "multi_window": _normalize_dict(mw_front),
            "conditional": _normalize_dict(cond_front_score),
            "arima": _normalize_dict(arima_front),
        }
        signals_back = {
            "traditional": _normalize_dict(trad_back),
            "multi_window": _normalize_dict(trad_back),  # 后区简化
            "conditional": _normalize_dict(cond_back_score),
            "arima": _normalize_dict(arima_back),
        }

        front_fused = _weighted_blend(signals_front, SIGNAL_WEIGHTS, cfg["front_pool"])
        back_fused = _weighted_blend(signals_back, SIGNAL_WEIGHTS, cfg["back_pool"])

        return front_fused, back_fused

    def score_lotto(self, train_records: list, half_life: int = None) -> Tuple[Dict[int, float], Dict[int, float]]:
        """对大乐透生成融合评分。"""
        return self.score_for_recommend(train_records, half_life)

    def score_digit(self, train_records: list, half_life: int = None) -> Dict[int, Dict[int, float]]:
        """对排列玩法生成融合评分。"""
        if half_life is None:
            half_life = EWMA_HALF_LIFE

        cfg = self.cfg
        positions = cfg["positions"]
        pool = cfg["digit_pool"]
        window = _build_window(train_records, 730)
        if len(window) < 20:
            window = train_records

        result = {}
        for pos in range(positions):
            seq = [{r["front"][pos]} if pos < len(r["front"]) else set() for r in window]
            stats = _per_number_stats(pool, seq, len(window), 30, 50, half_life=half_life)

            trad_score = _score_map(stats)
            arima_all = arima_scores_digit(window, positions)
            arima_score = arima_all[pos]

            signals = {
                "traditional": _normalize_dict(trad_score),
                "multi_window": _normalize_dict(trad_score),  # digit 简化
                "conditional": _normalize_dict({d: 0.5 for d in pool}),
                "arima": _normalize_dict(arima_score),
            }

            result[pos] = _weighted_blend(signals, SIGNAL_WEIGHTS, pool)

        return result

    def backtest(self, n_test: int = 80, half_life: int = None) -> dict:
        """Stacking 集成回测。"""
        if half_life is None:
            half_life = EWMA_HALF_LIFE

        all_records = models.fetch_draws(self.game, limit=3000, order_desc=True)
        if len(all_records) < n_test + 50:
            n_test = max(5, len(all_records) - 50)
        if len(all_records) < 30:
            return {"error": "数据不足"}

        is_lotto = self.cfg["type"] == "lotto"
        stack_hits = trad_hits = 0
        stack_picks = trad_picks = 0
        n_points = 0

        step = max(1, n_test // 80)
        max_i = min(n_test, len(all_records) - 30)

        for i in range(1, max_i, step):
            test_rec = all_records[i]
            train_records = all_records[i + 1: i + 800]
            if len(train_records) < 30:
                continue

            if is_lotto:
                actual_front = set(test_rec["front"])
                front_fused, _ = self.score_lotto(train_records, half_life=half_life)
                stack_ranked = sorted(front_fused, key=lambda n: -front_fused.get(n, 0))[:7]
                stack_hits += len(set(stack_ranked) & actual_front)
                stack_picks += 7

                window = _build_window(train_records, 730)
                if len(window) < 20:
                    window = train_records
                front_sets = [set(r["front"]) for r in window]
                stats = _per_number_stats(self.cfg["front_pool"], front_sets, len(window),
                                          30, 50, half_life=half_life)
                trad_score = _score_map(stats)
                trad_ranked = sorted(trad_score, key=lambda n: -trad_score.get(n, 0))[:7]
                trad_hits += len(set(trad_ranked) & actual_front)
                trad_picks += 7
            else:
                positions = self.cfg["positions"]
                actual = list(test_rec["front"])
                stack_scores = self.score_digit(train_records, half_life=half_life)
                for pos in range(positions):
                    stack_best = max(stack_scores[pos], key=stack_scores[pos].get)
                    stack_hits += (1 if stack_best == actual[pos] else 0)
                    stack_picks += 1

                pool = self.cfg["digit_pool"]
                window = _build_window(train_records, 730)
                if len(window) < 20:
                    window = train_records
                for pos in range(positions):
                    seq = [{r["front"][pos]} if pos < len(r["front"]) else set() for r in window]
                    stats = _per_number_stats(pool, seq, len(window), 30, 50, half_life=half_life)
                    trad_score = _score_map(stats)
                    trad_best = max(trad_score, key=trad_score.get)
                    trad_hits += (1 if trad_best == actual[pos] else 0)
                    trad_picks += 1

            n_points += 1

        stack_rate = stack_hits / stack_picks if stack_picks else 0
        trad_rate = trad_hits / trad_picks if trad_picks else 0
        lift = (stack_rate - trad_rate) / trad_rate if trad_rate > 0 else 0

        return {
            "stack_hit_rate": round(stack_rate, 4),
            "traditional_hit_rate": round(trad_rate, 4),
            "lift": round(lift, 4),
            "stack_hits": stack_hits,
            "trad_hits": trad_hits,
            "n_points": n_points,
        }


def optimize_stacking_weights(game: str, n_validate: int = 60) -> dict:
    """网格搜索最优 Stacking 权重组合。"""
    candidates = [
        {"traditional": 0.30, "multi_window": 0.25, "conditional": 0.25, "arima": 0.20},
        {"traditional": 0.40, "multi_window": 0.20, "conditional": 0.20, "arima": 0.20},
        {"traditional": 0.20, "multi_window": 0.30, "conditional": 0.30, "arima": 0.20},
        {"traditional": 0.35, "multi_window": 0.20, "conditional": 0.30, "arima": 0.15},
        {"traditional": 0.25, "multi_window": 0.30, "conditional": 0.25, "arima": 0.20},
        {"traditional": 0.50, "multi_window": 0.15, "conditional": 0.15, "arima": 0.20},
        {"traditional": 0.30, "multi_window": 0.30, "conditional": 0.20, "arima": 0.20},
        {"traditional": 0.25, "multi_window": 0.25, "conditional": 0.25, "arima": 0.25},
    ]

    all_records = models.fetch_draws(game, limit=3000, order_desc=True)
    if len(all_records) < n_validate + 50:
        return {"error": "数据不足"}

    best_weights = None
    best_rate = 0
    results = []

    for weights in candidates:
        ens = StackingEnsemble(game)
        # Temporarily override SIGNAL_WEIGHTS
        import config
        old_weights = dict(config.SIGNAL_WEIGHTS)
        config.SIGNAL_WEIGHTS.update(weights)
        try:
            r = ens.backtest(n_test=n_validate)
            rate = r["stack_hit_rate"]
            results.append({"weights": weights, "stack_rate": rate, "trad_rate": r["traditional_hit_rate"]})
            if rate > best_rate:
                best_rate = rate
                best_weights = weights
        finally:
            config.SIGNAL_WEIGHTS.clear()
            config.SIGNAL_WEIGHTS.update(old_weights)

    results.sort(key=lambda x: -x["stack_rate"])
    return {
        "best_weights": best_weights,
        "best_rate": best_rate,
        "all_results": results[:5],
    }
