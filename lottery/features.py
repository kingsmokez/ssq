# -*- coding: utf-8 -*-
"""增强特征工程 — 比单纯的频率/遗漏更稳健的统计信号。

设计动机：回测显示"直接选综合分最高的号"在大乐透上甚至比随机还差，
原因是模型把票全压在近期热号上（过度集中），而彩票有均值回归特性。
本模块提供更稳健的特征：

1. 贝叶斯平滑频率 (bayes_freq)
   freq = (出现次数 + C) / (总期数 + C*pool_size)
   C 越大越倾向均匀分布，避免小样本噪声把冷热拉到极端。

2. 指数加权移动平均频率 (ewma_freq)
   越近的期权重越高（half_life 控制），捕捉近期状态但比"近30期硬切"更平滑。

3. 遗漏分位 (omission_percentile)
   当前遗漏在历史遗漏分布中的百分位，比"遗漏比"更抗极值。

4. 号码共生倾向 (cooccur)
   lotto 前区号码间的同现率——用于推荐时避免/倾向某些组合。

⚠️ 这些特征让评分更稳健，但**不会让模型真的能预测**（回测已证明）。
   它们的价值是让推荐更"合理分散"，而不是反向。
"""

from typing import Dict, List


def bayes_smoothed_freq(appear_count: int, total: int, pool_size: int,
                        C: float = 5.0) -> float:
    """贝叶斯平滑频率。C=0 退化为普通频率；C 大→更均匀。"""
    if total <= 0:
        return 1.0 / pool_size if pool_size else 0.0
    return (appear_count + C) / (total + C * pool_size)


def ewma_freq(presence_seq: List[int], half_life: int = 25) -> float:
    """指数加权频率。presence_seq[0]=最近一期，1=出现 0=未出现。
    half_life: 权重衰减一半所需期数。
    """
    if not presence_seq:
        return 0.0
    alpha = 0.5 ** (1.0 / half_life) if half_life > 0 else 1.0
    weight_sum = 0.0
    hit_sum = 0.0
    w = 1.0
    for present in presence_seq:
        weight_sum += w
        hit_sum += w * (1 if present else 0)
        w *= alpha
    return hit_sum / weight_sum if weight_sum > 0 else 0.0


def omission_percentile(current_omission: int, omission_gaps: List[int]) -> float:
    """当前遗漏在历史遗漏段分布中的百分位 [0,1]。"""
    if not omission_gaps:
        return 0.5
    below = sum(1 for g in omission_gaps if g <= current_omission)
    return below / len(omission_gaps)


def cooccurrence_matrix(records_front: List[List[int]], pool: List[int]) -> Dict:
    """计算号码两两共生率。返回 {n: {m: co_rate}}。
    co_rate(n,m) = 两号同时出现期数 / n 出现期数。
    """
    pair_count = {n: {m: 0 for m in pool} for n in pool}
    single_count = {n: 0 for n in pool}
    for front in records_front:
        fset = set(front) & set(pool)
        for n in fset:
            single_count[n] += 1
        for n in fset:
            for m in fset:
                if m != n:
                    pair_count[n][m] += 1
    co = {n: {} for n in pool}
    for n in pool:
        if single_count[n] > 0:
            for m in pool:
                if m != n and pair_count[n][m] > 0:
                    co[n][m] = pair_count[n][m] / single_count[n]
    return co
