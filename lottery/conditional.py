# -*- coding: utf-8 -*-
"""条件概率推荐法 — 利用号码间条件概率 P(B|A) 做联合推荐。

核心思路：
  在大乐透中，当某个号码 A 出现时，号码 B 也出现的概率 P(B|A)
  可能与无条件概率 P(B) 不同（虽然差异很小，因为独立随机）。
  
  条件概率法的逻辑：
  1. 用历史数据估计 P(B|A) 对所有号码对
  2. 先选综合分最高的号码 A
  3. 然后选在 A 出现条件下最可能出现的 B
  4. 依次扩展到 5 个号
  
  反冷号策略：
  回测证明"冷号回补"是最差策略。反冷号策略直接排除
  高遗漏号，只从"近期出现过"的号码池中选号。
"""

from typing import Dict, List, Set
from collections import defaultdict

from config import GAMES
from .analyzer import _per_number_stats, _build_window


def conditional_prob_matrix(records: list, pool: list) -> Dict[int, Dict[int, float]]:
    """计算号码间条件概率 P(B|A) = P(A∧B) / P(A)。

    records: 开奖记录，最新在前
    pool: 号码池

    返回 {A: {B: P(B|A)}}
    """
    pair_count = defaultdict(lambda: defaultdict(int))
    single_count = defaultdict(int)

    for r in records:
        front = set(r.get("front", [])) & set(pool)
        for n in front:
            single_count[n] += 1
        for n in front:
            for m in front:
                if m != n:
                    pair_count[n][m] += 1

    cond_prob = {}
    for a in pool:
        if single_count.get(a, 0) > 0:
            cond_prob[a] = {}
            for b in pool:
                if b != a:
                    cond_prob[a][b] = pair_count[a].get(b, 0) / single_count[a]
    return cond_prob


def conditional_pick(stats: dict, score: dict, cond_prob: dict,
                     pool: list, k: int = 5) -> List[int]:
    """条件概率选号：先选最高分号，然后选条件概率最高的号。

    策略：
    1. 选综合分最高的号 A
    2. 在剩余号中，选 P(B|A) 最高的 B
    3. 在剩余号中，选 max(P(C|A)*P(C|B)) 的 C
    4. 继续扩展到 k 个号
    """
    picked = []
    remaining = set(pool)

    # 第1个：综合分最高
    best = max(remaining, key=lambda n: score.get(n, 0))
    picked.append(best)
    remaining.discard(best)

    # 后续：在已选号条件下，选条件概率最高的
    while len(picked) < k and remaining:
        best_next = None
        best_cond = -1
        for n in remaining:
            # 对所有已选号的条件概率取平均
            cond_avg = 0.0
            count = 0
            for p in picked:
                cp = cond_prob.get(p, {}).get(n, 0)
                cond_avg += cp
                count += 1
            if count > 0:
                cond_avg /= count
            # 综合分和条件概率加权
            combined = 0.5 * score.get(n, 0) + 0.5 * cond_avg
            if combined > best_cond:
                best_cond = combined
                best_next = n

        if best_next is not None:
            picked.append(best_next)
            remaining.discard(best_next)
        else:
            break

    return sorted(picked)


def anti_cold_pick(stats: dict, score: dict, pool: list,
                   k: int = 5, max_omission_ratio: float = 1.5) -> List[int]:
    """反冷号策略：排除高遗漏号，只从"近期活跃"的号码中选。

    max_omission_ratio: 遗漏比超过此值的号码被排除

    回测发现：冷号回补是全局最差策略（遗漏比>1.5的号命中率低于随机）。
    反冷号策略直接排除这些号。
    """
    # 筛选活跃号码
    active = [n for n in pool
              if stats.get(n) and stats[n].omission_ratio <= max_omission_ratio]

    # 如果活跃号码不够，逐步放宽
    while len(active) < k + 2:
        max_omission_ratio += 0.5
        active = [n for n in pool
                  if stats.get(n) and stats[n].omission_ratio <= max_omission_ratio]
        if max_omission_ratio > 10:
            active = list(pool)
            break

    # 在活跃号中按综合分选
    ranked = sorted(active, key=lambda n: -score.get(n, 0))
    return sorted(ranked[:k])


def hybrid_pick(stats: dict, score: dict, cond_prob: dict,
                pool: list, k: int = 5) -> List[int]:
    """混合策略：反冷号 + 条件概率。

    1. 先用反冷号过滤出活跃号池
    2. 在活跃号池中用条件概率选号
    """
    # 活跃号池
    active = [n for n in pool
              if stats.get(n) and stats[n].omission_ratio <= 1.5]
    if len(active) < k + 2:
        active = [n for n in pool
                  if stats.get(n) and stats[n].omission_ratio <= 2.5]
    if len(active) < k + 2:
        active = list(pool)

    # 条件概率选号
    picked = []
    remaining = set(active)

    best = max(remaining, key=lambda n: score.get(n, 0))
    picked.append(best)
    remaining.discard(best)

    while len(picked) < k and remaining:
        best_next = None
        best_combined = -1
        for n in remaining:
            cond_avg = 0.0
            count = 0
            for p in picked:
                cp = cond_prob.get(p, {}).get(n, 0)
                cond_avg += cp
                count += 1
            if count > 0:
                cond_avg /= count
            combined = 0.5 * score.get(n, 0) + 0.5 * cond_avg
            if combined > best_combined:
                best_combined = combined
                best_next = n
        if best_next is not None:
            picked.append(best_next)
            remaining.discard(best_next)
        else:
            break

    return sorted(picked)
