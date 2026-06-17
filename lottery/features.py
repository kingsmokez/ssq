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


def markov_transition_matrix(records_digits: List[List[int]], positions: int,
                              pool: List[int] = None) -> Dict[int, List[List[float]]]:
    """计算 digit 玩法每位的马尔可夫转移矩阵。

    records_digits: 每期各位数字列表，最新在前。
    positions: 位数（3 或 5）
    pool: 可选数字池（默认 0-9）

    返回: {pos: T[pos]} where T[i][j] = P(下期数字=j | 本期数字=i)
    注意：每期之间独立，所以"本期→下期"是对相邻期（t→t+1）的转移概率。

    如果开奖是真正随机的，则 T[i][j] ≈ 0.1 对所有 i,j 均成立。
    """
    if pool is None:
        pool = list(range(10))
    n_digits = len(pool)

    # 初始化转移计数矩阵
    trans_count = {}
    for pos in range(positions):
        trans_count[pos] = [[0] * n_digits for _ in range(n_digits)]

    # 统计相邻期之间的转移：t → t+1（records 最新在前，即 t-1 → t）
    # 从旧到新遍历
    for idx in range(len(records_digits) - 1, 0, -1):
        prev_digits = records_digits[idx]      # 较早的期
        curr_digits = records_digits[idx - 1]  # 较新的期（下一期）
        for pos in range(positions):
            if (pos < len(prev_digits) and pos < len(curr_digits)
                    and prev_digits[pos] in pool and curr_digits[pos] in pool):
                from_idx = pool.index(prev_digits[pos])
                to_idx = pool.index(curr_digits[pos])
                trans_count[pos][from_idx][to_idx] += 1

    # 转为概率
    trans_prob = {}
    for pos in range(positions):
        tprob = []
        for from_idx in range(n_digits):
            row_total = sum(trans_count[pos][from_idx])
            if row_total > 0:
                tprob.append([trans_count[pos][from_idx][to_idx] / row_total
                              for to_idx in range(n_digits)])
            else:
                tprob.append([1.0 / n_digits] * n_digits)
        trans_prob[pos] = tprob
    return trans_prob


def markov_anomaly(trans_prob: Dict[int, List[List[float]]]) -> dict:
    """检测马尔可夫转移矩阵中的异常（偏差最大的转移对）。

    返回每位偏差最大的 top-3 转移对及其偏差倍率。
    """
    anomalies = {}
    for pos, T in trans_prob.items():
        n = len(T)
        uniform = 1.0 / n
        devs = []
        for i in range(n):
            for j in range(n):
                dev = T[i][j] / uniform if uniform > 0 else 1.0
                if abs(dev - 1.0) > 0.3:  # 偏差 > 30% 的才记录
                    devs.append((i, j, round(dev, 3), round(T[i][j], 4)))
        devs.sort(key=lambda x: -abs(x[2] - 1.0))
        anomalies[pos] = devs[:3]
    return anomalies


# ---------------- 间距(Gap)分布分析 ----------------

def gap_distribution(front_sets: List[List[int]], max_num: int = 35) -> Dict[str, dict]:
    """分析大乐透前区排序后号码间距(gap)的分布。

    对5个排序号码 n1<n2<n3<n4<n5，定义4个间距：
      gap1=n2-n1, gap2=n3-n2, gap3=n4-n3, gap4=n5-n4

    返回每个gap的分布统计数据，用于：
      - 验证组合间距是否"正常"
      - 过滤掉间距极端的异常组合
    """
    if not front_sets:
        return {}

    gaps = {"gap1": [], "gap2": [], "gap3": [], "gap4": []}
    for front in front_sets:
        s = sorted(front)
        for i in range(4):
            gap_key = f"gap{i+1}"
            gaps[gap_key].append(s[i+1] - s[i])

    result = {}
    for key, vals in gaps.items():
        if not vals:
            continue
        vals.sort()
        n = len(vals)
        result[key] = {
            "mean": round(sum(vals) / n, 2),
            "median": vals[n // 2],
            "min": vals[0],
            "max": vals[-1],
            "p10": vals[int(n * 0.1)],
            "p25": vals[int(n * 0.25)],
            "p75": vals[int(n * 0.75)],
            "p90": vals[int(n * 0.9)],
        }
    return result


def gap_score(front_combo: List[int], gap_dist: Dict[str, dict]) -> float:
    """给一个前区组合的间距合规度打分（0~1）。

    对每个 gap，检查是否落在历史分布的 [p10, p90] 区间内。
    全部 4 个 gap 都合规 → 1.0，都不合规 → 0.0。
    可用于 _constrain_front 的额外加分项。
    """
    if not gap_dist:
        return 0.5
    s = sorted(front_combo)
    score = 0.0
    for i in range(4):
        key = f"gap{i+1}"
        if key not in gap_dist:
            continue
        g = s[i+1] - s[i]
        dist = gap_dist[key]
        if dist["p10"] <= g <= dist["p90"]:
            score += 0.25  # 每个gap贡献0.25
    return score


# ---------------- 星期几效应分析 ----------------

def weekday_distribution(draws, pool=None) -> dict:
    """按星期几分段统计号码出现频率。

    draws: 开奖记录列表（最新在前），每条含 draw_date 和 front。

    返回 {weekday(0=Mon): {total_draws, hot, cold}}
    """
    from collections import Counter
    from datetime import datetime

    weekday_stats = {d: {"total_draws": 0, "counts": Counter()}
                     for d in range(7)}

    for r in draws:
        date_str = r.get("draw_date", "")
        if not date_str:
            continue
        try:
            wd = datetime.strptime(date_str[:10], "%Y-%m-%d").weekday()
        except ValueError:
            continue
        weekday_stats[wd]["total_draws"] += 1
        for n in r.get("front", []):
            weekday_stats[wd]["counts"][n] += 1

    result = {}
    for wd in range(7):
        st = weekday_stats[wd]
        if st["total_draws"] == 0:
            continue
        ranked = sorted(st["counts"], key=lambda n: -st["counts"][n])
        result[wd] = {
            "total_draws": st["total_draws"],
            "hot": [(n, round(st["counts"][n]/st["total_draws"], 3)) for n in ranked[:3]],
            "cold": [(n, round(st["counts"][n]/st["total_draws"], 3)) for n in ranked[-3:]],
        }
    return result
