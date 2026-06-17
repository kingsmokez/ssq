# -*- coding: utf-8 -*-
"""多窗口特征集成 — 用不同时间窗口的特征做加权融合。

设计动机：不同统计特征在不同时间尺度上可能有不同的预测能力。
  - 短期窗口 (90天)：捕捉近期动量，反应快但噪声大
  - 中期窗口 (365天)：平衡信号和噪声
  - 长期窗口 (730天)：稳定但有滞后

通过训练一个元模型来学习最优窗口权重组合。
"""

import numpy as np
from typing import Dict, List

from config import GAMES, WINDOW_CONFIGS as _CFG_WINDOW_CONFIGS
from .analyzer import _per_number_stats, _build_window


# 使用 config.py 中的统一窗口配置（含权重）
WINDOW_CONFIGS = _CFG_WINDOW_CONFIGS


def multi_window_features(game: str, records: list,
                           half_life: int = 25) -> dict:
    """为某游戏的号码计算多窗口特征。

    返回 {number: {window_name: {feature_name: value}}} 的三层嵌套结构。
    """
    cfg = GAMES[game]
    pool = cfg["front_pool"] if cfg["type"] == "lotto" else cfg["digit_pool"]

    all_features = {}
    for wc in WINDOW_CONFIGS:
        window = _build_window(records, wc["days"])
        if len(window) < 10:
            window = records[: max(10, len(records))]

        seq = [set(r["front"]) for r in window]
        stats = _per_number_stats(
            pool, seq, len(window),
            wc["momentum_n"], wc["cold_n"],
            half_life=half_life)

        for n in pool:
            s = stats.get(n)
            if s is None:
                continue
            if n not in all_features:
                all_features[n] = {}
            all_features[n][wc["name"]] = {
                "freq": s.freq,
                "omission": s.omission,
                "momentum": s.momentum,
                "ewma_freq": s.ewma_freq,
                "omission_pct": s.omission_pct,
                "cold_hot": s.cold_hot,
            }

    return all_features


def ensemble_multi_window_score(all_features: dict,
                                 window_weights: dict = None) -> Dict[int, float]:
    """多窗口特征加权融合为一个综合评分。

    window_weights: 各窗口权重，默认等权。
    返回 {number: score}
    """
    if window_weights is None:
        window_weights = {wc["name"]: wc.get("weight", 0.25) for wc in WINDOW_CONFIGS}

    scores = {}
    for n, window_feats in all_features.items():
        total = 0.0
        weight_sum = 0.0
        for wn, wf in window_feats.items():
            w = window_weights.get(wn, 0.25)
            # 综合该窗口的各项特征
            s = (wf["freq"] * 0.4 +
                 (1 - wf["omission_pct"]) * 0.3 +
                 max(0, wf["momentum"]) * 0.2 +
                 wf["ewma_freq"] * 0.1)
            total += s * w
            weight_sum += w
        scores[n] = total / weight_sum if weight_sum > 0 else 0.0

    return scores


def optimize_window_weights(game: str, n_validate: int = 50) -> dict:
    """用网格搜索找最优窗口权重组合。

    在验证集上测试不同的权重组合，返回最优组合。
    """
    from . import models
    from .backtester import _build_window as bw

    all_records = models.fetch_draws(game, limit=2000, order_desc=True)
    if len(all_records) < n_validate + 100:
        return {"error": "数据不足"}

    # 权重候选
    candidates = [
        {"short": 0.25, "medium": 0.25, "long": 0.25, "very_long": 0.25},  # 等权
        {"short": 0.40, "medium": 0.30, "long": 0.20, "very_long": 0.10},  # 偏短期
        {"short": 0.10, "medium": 0.20, "long": 0.40, "very_long": 0.30},  # 偏长期
        {"short": 0.30, "medium": 0.40, "long": 0.20, "very_long": 0.10},  # 偏中期
        {"short": 0.50, "medium": 0.20, "long": 0.20, "very_long": 0.10},  # 强偏短期
        {"short": 0.10, "medium": 0.10, "long": 0.30, "very_long": 0.50},  # 强偏长期
    ]

    is_lotto = GAMES[game]["type"] == "lotto"
    pool = GAMES[game]["front_pool"] if is_lotto else GAMES[game]["digit_pool"]

    best_weights = None
    best_hits = 0
    results = []

    for weights in candidates:
        total_hits = 0
        total_picks = 0

        step = max(1, n_validate // 50)
        for i in range(1, min(n_validate, len(all_records) - 30), step):
            test_rec = all_records[i]
            train_records = all_records[i + 1: i + 500]
            if len(train_records) < 30:
                continue

            feats = multi_window_features(game, train_records)
            scores = ensemble_multi_window_score(feats, weights)
            ranked = sorted(scores, key=lambda n: -scores.get(n, 0))

            k = min(7, len(ranked))
            actual = set(test_rec["front"]) if is_lotto else set(test_rec["front"])
            hits = len(set(ranked[:k]) & actual)
            total_hits += hits
            total_picks += k

        rate = total_hits / total_picks if total_picks else 0
        results.append({"weights": weights, "hit_rate": rate, "hits": total_hits})

        if total_hits > best_hits:
            best_hits = total_hits
            best_weights = weights

    return {
        "best_weights": best_weights,
        "best_rate": round(best_hits / max(1, total_picks), 4) if best_weights else 0,
        "all_results": sorted(results, key=lambda x: -x["hit_rate"]),
    }
