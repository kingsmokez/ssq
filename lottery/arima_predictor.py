# -*- coding: utf-8 -*-
"""ARIMA 时间序列预测 — 对每个号码的出现频率做自回归预测。

核心思路：
  把每个号码的历史出现建模为时间序列（每期出现1/未出现0），
  用自回归模型 AR(p) 预测下一期出现的概率。

  简单但有效：AR(1) 只需要估计一个参数 ρ（自相关系数），
  然后预测 = ρ * 上一期状态 + (1-ρ) * 均值。

  这比手工评分更"时间序列原生"——它直接建模序列依赖，
  而不是把特征堆在一起做线性加权。
"""

import math
from typing import Dict, List
import numpy as np

from config import GAMES
from . import models


def _build_presence_series(records: list, target_number: int) -> List[int]:
    """为某个号码构建出现/未出现序列（最新在前）。

    返回 [latest, ..., oldest]，每个元素 1=出现 0=未出现。
    """
    series = []
    for r in records:
        front = r.get("front", [])
        series.append(1 if target_number in front else 0)
    return series


def ar1_fit_predict(series: List[int]) -> float:
    """AR(1) 模型：估计自相关系数 ρ 并预测下一期概率。

    X_t = ρ * X_{t-1} + (1-ρ) * μ + ε_t

    μ = 均值（长期频率）
    ρ = Corr(X_t, X_{t-1})（一阶自相关）

    返回 predicted P(X_next = 1)
    """
    if len(series) < 2:
        return 0.0

    arr = np.array(series, dtype=float)
    mu = np.mean(arr)  # 长期均值（频率）

    # 计算一阶自相关系数
    # ρ = Cov(X_t, X_{t-1}) / Var(X)
    x_curr = arr[:-1]  # t 期
    x_prev = arr[1:]   # t-1 期

    var_x = np.var(arr)
    if var_x < 1e-9:
        return mu

    cov = np.mean((x_curr - mu) * (x_prev - mu))
    rho = cov / var_x
    rho = max(-0.9, min(0.9, rho))  # 裁剪到合理范围

    # 预测：X_next = ρ * X_latest + (1-ρ) * μ
    latest = arr[0]  # 最近一期
    prediction = rho * latest + (1 - rho) * mu
    return float(max(0.0, min(1.0, prediction)))


def arima_scores_lotto(records: list, pool: list) -> Dict[int, float]:
    """对乐透号码池中每个号码做 AR(1) 预测。

    records: 开奖记录，最新在前
    pool: 号码池

    返回 {number: predicted_probability}
    """
    scores = {}
    for n in pool:
        series = _build_presence_series(records, n)
        prob = ar1_fit_predict(series)
        scores[n] = prob
    return scores


def arima_scores_digit(records: list, positions: int) -> Dict[int, Dict[int, float]]:
    """对排列玩法的每位做 AR(1) 预测。

    返回 {pos: {digit: probability}}
    """
    result = {}
    for pos in range(positions):
        pos_scores = {}
        for d in range(10):
            # 构建该位置该数字的出现序列
            series = []
            for r in records:
                digits = r.get("front", [])
                if pos < len(digits):
                    series.append(1 if digits[pos] == d else 0)
                else:
                    series.append(0)
            prob = ar1_fit_predict(series)
            pos_scores[d] = prob
        result[pos] = pos_scores
    return result


def arima_metrics(series: List[int]) -> dict:
    """计算 AR(1) 模型的拟合质量指标。

    返回 {rho, mu, prediction, residual_var, aic_approx}
    """
    if len(series) < 3:
        return {"rho": 0, "mu": 0, "prediction": 0}

    arr = np.array(series, dtype=float)
    mu = float(np.mean(arr))

    x_curr = arr[:-1]
    x_prev = arr[1:]
    var_x = np.var(arr)
    if var_x < 1e-9:
        rho = 0.0
    else:
        cov = np.mean((x_curr - mu) * (x_prev - mu))
        rho = float(cov / var_x)
        rho = max(-0.9, min(0.9, rho))

    latest = arr[0]
    prediction = rho * latest + (1 - rho) * mu

    # 残差方差
    residuals = x_curr - (rho * x_prev + (1 - rho) * mu)
    residual_var = float(np.var(residuals))

    return {
        "rho": round(rho, 4),
        "mu": round(mu, 4),
        "prediction": round(float(prediction), 4),
        "residual_var": round(residual_var, 6),
        "latest": int(latest),
    }


def arima_backtest(game: str, n_test: int = 50) -> dict:
    """用 AR(1) 模型做逐期回测，与传统评分对比。

    返回 {arima_hit_rate, traditional_hit_rate, lift}
    """
    from .recommender import _score_map
    from .analyzer import _per_number_stats, _build_window
    from .backtester import _build_window as bw

    cfg = GAMES[game]
    is_lotto = cfg["type"] == "lotto"

    all_records = models.fetch_draws(game, limit=3000, order_desc=True)
    if len(all_records) < n_test + 50:
        n_test = max(5, len(all_records) - 50)
    if len(all_records) < 30:
        return {"error": "数据不足"}

    ar_hits = trad_hits = 0
    ar_picks = trad_picks = 0
    n_points = 0

    step = max(1, n_test // 80)
    max_i = min(n_test, len(all_records) - 30)

    for i in range(1, max_i, step):
        test_rec = all_records[i]
        train_records = all_records[i + 1: i + 800]
        if len(train_records) < 30:
            continue
        train_window = bw(train_records, 730)
        if len(train_window) < 20:
            train_window = train_records

        if is_lotto:
            actual_front = set(test_rec["front"])
            pool = cfg["front_pool"]

            # AR(1) 评分
            ar_scores = arima_scores_lotto(train_window, pool)
            ar_ranked = sorted(ar_scores, key=lambda n: -ar_scores.get(n, 0))[:7]
            ar_hits += len(set(ar_ranked) & actual_front)
            ar_picks += 7

            # 传统评分
            front_sets = [set(r["front"]) for r in train_window]
            stats = _per_number_stats(pool, front_sets, len(train_window), 30, 50)
            trad_score = _score_map(stats)
            trad_ranked = sorted(trad_score, key=lambda n: -trad_score.get(n, 0))[:7]
            trad_hits += len(set(trad_ranked) & actual_front)
            trad_picks += 7
        else:
            positions = cfg["positions"]
            actual = list(test_rec["front"])

            ar_scores = arima_scores_digit(train_window, positions)
            for pos in range(positions):
                ar_best = max(ar_scores[pos], key=ar_scores[pos].get)
                ar_hits += (1 if ar_best == actual[pos] else 0)
                ar_picks += 1

            pool = cfg["digit_pool"]
            for pos in range(positions):
                seq = []
                for r in train_window:
                    digits = r["front"]
                    seq.append({digits[pos]} if pos < len(digits) else set())
                stats = _per_number_stats(pool, seq, len(train_window), 30, 50)
                trad_score = _score_map(stats)
                trad_best = max(trad_score, key=trad_score.get)
                trad_hits += (1 if trad_best == actual[pos] else 0)
                trad_picks += 1

        n_points += 1

    ar_rate = ar_hits / ar_picks if ar_picks else 0
    trad_rate = trad_hits / trad_picks if trad_picks else 0
    lift = (ar_rate - trad_rate) / trad_rate if trad_rate > 0 else 0

    return {
        "arima_hit_rate": round(ar_rate, 4),
        "traditional_hit_rate": round(trad_rate, 4),
        "lift": round(lift, 4),
        "ar_hits": ar_hits,
        "trad_hits": trad_hits,
        "n_points": n_points,
    }
