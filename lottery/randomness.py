# -*- coding: utf-8 -*-
"""随机性检测 — 用统计检验验证历史开奖是否真的是均匀随机。

设计动机：如果有人怀疑"彩票有规律/被操纵"，随机性检测可以客观地回答：
  "目前的开奖数据在统计上是否符合均匀随机假设？"

所有检验的 H0：数据是独立同分布的均匀随机。
"""

import math
from collections import Counter
from typing import Dict, List


def _normal_cdf(z: float) -> float:
    """标准正态分布的 CDF。"""
    if z < -8:
        return 0.0
    if z > 8:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989422804014327
    p = d * math.exp(-z * z / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (
            1.781477937 + t * (-1.821255978 + t * 1.330274429)))))
    return 1.0 - p if z > 0 else p


def chi_square_uniform(observed_counts: Dict[int, int], expected_per: float) -> dict:
    """卡方拟合优度检验：观测分布是否与均匀分布一致。

    observed_counts: {号码: 出现次数}
    expected_per: 均匀分布下每个号的期望次数

    返回 {chi2, p_value, df, is_significant, verdict}
    """
    if not observed_counts or expected_per <= 0:
        return {"chi2": 0, "p_value": 1.0, "df": 0,
                "is_significant": False,
                "verdict": "数据不足，无法检验。"}

    n_categories = len(observed_counts)
    chi2 = 0.0
    for n in observed_counts:
        obs = observed_counts[n]
        chi2 += (obs - expected_per) ** 2 / expected_per

    df = n_categories - 1
    # 卡方分布 p 值近似
    if chi2 < 0.01 or df <= 0:
        p_val = 1.0
    else:
        # 用正态近似：sqrt(2*chi2) - sqrt(2*df-1) ≈ N(0,1)
        z = math.sqrt(2 * chi2) - math.sqrt(2 * df - 1)
        p_val = 2 * (1 - _normal_cdf(abs(z)))  # 双侧
    p_val = max(0.0, min(1.0, p_val))

    # 判断
    significant = p_val < 0.05
    if significant:
        verdict = f"p={p_val:.4f} < 0.05，统计显著：观测分布与均匀分布有差异，但需注意多重比较问题。"
    else:
        verdict = f"p={p_val:.4f} > 0.05，统计不显著：数据与均匀分布一致。"

    return {"chi2": round(chi2, 4), "p_value": round(p_val, 4),
            "df": df, "is_significant": significant, "verdict": verdict}


def runs_test(sequence: List[int], median: float = None) -> dict:
    """Runs Test（游程检验）：检测序列是否具有趋势或周期性。

    sequence: 数字序列
    median: 分割点（默认用中位数）

    返回 {n_runs, n1, n2, expected_runs, std_runs, z, p_value, verdict}
    """
    if len(sequence) < 10:
        return {"n_runs": 0, "p_value": 1.0, "verdict": "序列太短，无法检验。"}

    if median is None:
        sorted_seq = sorted(sequence)
        median = sorted_seq[len(sorted_seq) // 2]

    # 将序列转换为二元：大于中位数为 1，否则为 0
    binary = [1 if x > median else 0 for x in sequence]
    n1 = sum(binary)          # > 中位数的个数
    n2 = len(binary) - n1     # ≤ 中位数的个数
    if n1 == 0 or n2 == 0:
        return {"n_runs": 1, "n1": n1, "n2": n2,
                "expected_runs": 1, "z": 0, "p_value": 1.0,
                "verdict": "数据全在同一侧，无法判断。"}

    # 计算游程数（连续的相同值段数）
    n_runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i - 1]:
            n_runs += 1

    # 期望游程数
    N = n1 + n2
    expected_runs = 2 * n1 * n2 / (n1 + n2) + 1
    std_runs = math.sqrt((2 * n1 * n2 * (2 * n1 * n2 - N)) / (N * N * (N - 1)))
    if std_runs < 1e-9:
        z = 0.0
        p_val = 1.0
    else:
        # 连续性校正
        z = (n_runs - expected_runs - 0.5) / std_runs if n_runs > expected_runs else \
            (n_runs - expected_runs + 0.5) / std_runs
        p_val = 2 * (1 - _normal_cdf(abs(z)))

    p_val = max(0.0, min(1.0, p_val))
    significant = p_val < 0.05

    if significant:
        if n_runs < expected_runs:
            verdict = f"游程数({n_runs})显著少于期望({expected_runs:.1f})，p={p_val:.4f}，可能存在聚集趋势（号码聚集出现）。"
        else:
            verdict = f"游程数({n_runs})显著多于期望({expected_runs:.1f})，p={p_val:.4f}，可能存在交替模式。"
    else:
        verdict = f"游程数({n_runs})与期望({expected_runs:.1f})一致，p={p_val:.4f}，序列随机性无异常。"

    return {"n_runs": n_runs, "n1": n1, "n2": n2,
            "expected_runs": round(expected_runs, 2),
            "std_runs": round(std_runs, 4),
            "z": round(z, 4), "p_value": round(p_val, 4),
            "is_significant": significant, "verdict": verdict}


def frequency_balance(observed_counts: Dict[int, int], total_draws: int,
                      pool_size: int) -> dict:
    """频率平衡分析：每个号码的实际出现频率 vs 期望频率。

    observed_counts: {号码: 出现次数}
    total_draws: 总期数
    pool_size: 号池大小

    返回 {max_freq, min_freq, expected_freq, max_deviation_pct, ...}
    """
    if total_draws <= 0 or pool_size <= 0:
        return {"error": "数据不足"}

    expected = total_draws / pool_size
    max_dev = 0.0
    max_dev_num = None
    min_freq = float('inf')
    max_freq = 0.0

    for n in range(1, pool_size + 1):
        obs = observed_counts.get(n, 0)
        dev = (obs - expected) / expected * 100  # 百分比偏差
        if abs(dev) > abs(max_dev):
            max_dev = dev
            max_dev_num = n
        if obs < min_freq:
            min_freq = obs
        if obs > max_freq:
            max_freq = obs

    return {
        "expected_freq": round(expected, 2),
        "min_freq": min_freq,
        "max_freq": max_freq,
        "max_deviation_num": max_dev_num,
        "max_deviation_pct": round(max_dev, 2),
        "max_deviation_direction": "偏热" if max_dev > 0 else "偏冷",
    }


def test_draws(draws: List[dict], pool_size: int = 35, draw_pick: int = 5) -> dict:
    """对一组开奖记录运行全套随机性检测。

    draws: 开奖记录列表（最新在前）
    pool_size: 号池大小
    draw_pick: 每期开几个号

    返回 {chi_square, runs_test, frequency_balance, overall}
    """
    if not draws:
        return {"error": "无数据"}

    # 统计每个号码的出现次数
    counts = Counter()
    sequence = []  # 用于游程检验的序列
    for r in draws:
        front = r.get("front", [])
        for n in front:
            counts[n] += 1
        if front:
            sequence.append(sum(front))  # 用和值作为序列

    total = len(draws)
    total_appear = sum(counts.values())
    expected_per = total_appear / pool_size if pool_size else 1

    # 补齐未出现的号码
    for n in range(1, pool_size + 1):
        if n not in counts:
            counts[n] = 0

    chi2_res = chi_square_uniform(dict(counts), expected_per)
    runs_res = runs_test(sequence)
    bal_res = frequency_balance(dict(counts), total_appear, pool_size)

    # 综合结论
    n_significant = sum([
        1 for r in [chi2_res, runs_res]
        if r.get("is_significant", False)
    ])
    if n_significant >= 2:
        overall = ("多项检验提示统计显著，但需注意：(1) 彩票独立随机是数学定理，"
                   "统计显著 ≠ 可预测；(2) 多重比较下偶发显著是正常的。")
    elif n_significant == 1:
        overall = ("有一项检验提示统计显著，但单一检验在 5% 显著性水平下"
                   "每 20 次就会出现 1 次假阳性，不必过度解读。")
    else:
        overall = ("所有随机性检验均不显著(p>0.05)，"
                   "数据与均匀随机分布一致。不存在可被历史统计识别的规律。")

    return {
        "chi_square": chi2_res,
        "runs_test": runs_res,
        "frequency_balance": bal_res,
        "total_draws": total,
        "total_appearances": total_appear,
        "pool_size": pool_size,
        "overall": overall,
    }
