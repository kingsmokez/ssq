# -*- coding: utf-8 -*-
"""推荐引擎 v4 — 位置感知 + 多信号融合 + 反冷号 + 增强多样性 + 真集成投票。

v4 核心改进：
1. 位置感知评分：大乐透前区5个位置独立评分后组合
2. 多信号融合：传统+多窗口+条件概率+ARIMA
3. 反冷号策略：替换原冷号回补
4. 增强多样性惩罚：50%基础+75%重复
5. 真集成投票：各组加权投票
6. _score_map纯三维度：条件概率独立计算
"""

import random
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

from config import (
    WEIGHTS, GAMES, COMBO_MODES, DIGIT_COMBO_TOP_K, BET_UNIT_PRICE,
    SIGNAL_WEIGHTS, ANTI_COLD_MAX_OMISSION_RATIO,
    DIVERSITY_BASE_PENALTY, DIVERSITY_REPEAT_PENALTY,
    DIVERSITY_HARD_EXCLUDE_THRESHOLD,
    ENSEMBLE_VOTE_WEIGHT_DECAY, ENSEMBLE_MIN_VOTES,
    EWMA_HALF_LIFE, STATS_WINDOW_DAYS,
    POSITION_AWARE_TOP_N, POSITION_AWARE_WEIGHT,
)
from .analyzer import GameStats, NumberStat, analyze, _per_number_stats
from .explainer import explain_group, explain_combo


# ================ 评分工具 ================

def _norm(values):
    if not values:
        return {}
    vals = list(values.values())
    mean_val = sum(vals) / len(vals)
    variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
    std_val = variance ** 0.5
    if std_val < 1e-9:
        return {k: 0.5 for k in values}
    return {k: min(1.0, max(0.0, (v - mean_val) / (3 * std_val) * 0.5 + 0.5))
            for k, v in values.items()}


def _score_map(stats):
    """传统综合评分（v4 纯三维度：频率+遗漏+趋势）。

    条件概率不再嵌入此函数，由 _fused_score 独立计算，
    避免在传统信号中引入0.5噪声。
    """
    ewma = {n: getattr(s, "ewma_freq", s.freq) for n, s in stats.items()}
    bayes = {n: getattr(s, "bayes_freq", s.freq) for n, s in stats.items()}
    omp = {n: getattr(s, "omission_pct", 0.5) for n, s in stats.items()}
    mom = {n: s.momentum for n, s in stats.items()}

    ne, nb, no, nm = _norm(ewma), _norm(bayes), _norm(omp), _norm(mom)

    w = WEIGHTS
    return {
        n: (w["freq"] * 0.6 * ne.get(n, 0) + w["freq"] * 0.4 * nb.get(n, 0))
        + w["omission"] * no.get(n, 0)
        + w["momentum"] * 0.6 * nm.get(n, 0)
        for n in stats
    }


def _fused_score_lotto(gs, records=None):
    """大乐透多信号融合评分（v4 含位置感知）。"""
    cfg = GAMES[gs.game]
    front_pool = cfg["front_pool"]
    back_pool = cfg["back_pool"]

    trad_front = _score_map(gs.front_stats)
    trad_back = _score_map(gs.back_stats)

    mw_front = _multi_window_score(gs.front_stats, gs)
    mw_back = _multi_window_score(gs.back_stats, gs)

    cond_front = _conditional_score(gs.front_stats, gs.cooccur, front_pool)
    cond_back = _conditional_score(gs.back_stats, {}, back_pool)

    arima_front = _arima_score_lotto(gs, records, front_pool)
    arima_back = _arima_score_lotto(gs, records, back_pool, zone="back")

    signals_front = {
        "traditional": _normalize_dict(trad_front),
        "multi_window": _normalize_dict(mw_front),
        "conditional": _normalize_dict(cond_front),
        "arima": _normalize_dict(arima_front),
    }
    signals_back = {
        "traditional": _normalize_dict(trad_back),
        "multi_window": _normalize_dict(mw_back),
        "conditional": _normalize_dict(cond_back),
        "arima": _normalize_dict(arima_back),
    }

    front_fused = _weighted_blend(signals_front, SIGNAL_WEIGHTS, front_pool)
    back_fused = _weighted_blend(signals_back, SIGNAL_WEIGHTS, back_pool)

    # v4: 位置感知评分融合
    if records and len(records) >= 20:
        pos_scores = _position_aware_score(gs, records, front_pool)
        front_fused = _blend_position_aware(front_fused, pos_scores, front_pool)

    return front_fused, back_fused


def _fused_score_digit(gs, records=None):
    """排列玩法多信号融合评分。"""
    positions = len(gs.position_stats)
    pool = GAMES[gs.game]["digit_pool"]
    result = {}

    for pos in range(positions):
        stats = gs.position_stats[pos]
        trad_score = _score_map(stats)
        mw_score = _multi_window_score(stats, gs)
        cond_score = _conditional_score_digit(stats, gs, pos, records)
        arima_score = _arima_score_digit(gs, records, pos)

        signals = {
            "traditional": _normalize_dict(trad_score),
            "multi_window": _normalize_dict(mw_score),
            "conditional": _normalize_dict(cond_score),
            "arima": _normalize_dict(arima_score),
        }
        result[pos] = _weighted_blend(signals, SIGNAL_WEIGHTS, pool)

    return result


# ================ 位置感知评分 ================

def _position_aware_score(gs, records, pool):
    """大乐透前区按位评分：5个位置各自独立统计。

    在排序好的5个号码中，位置1更可能小、位置5更可能大。
    按位建模后，从每位各自的候选池中选号，比从35个整体池选号更精确。
    """
    if not records or len(records) < 20:
        return {}

    positions = 5
    result = {}

    for pos in range(positions):
        pos_seqs = []
        for r in records:
            front = r.get("front", [])
            if len(front) >= positions:
                sorted_front = sorted(front)
                pos_seqs.append({sorted_front[pos]})
            else:
                pos_seqs.append(set())

        pos_stats = _per_number_stats(pool, pos_seqs, len(pos_seqs),
                                       30, 50, half_life=EWMA_HALF_LIFE)
        pos_score = _score_map(pos_stats)
        result[pos] = _normalize_dict(pos_score)

    return result


def _blend_position_aware(overall_score, pos_scores, pool):
    """将整体评分与位置感知评分融合。"""
    if not pos_scores:
        return overall_score

    w = POSITION_AWARE_WEIGHT
    result = {}

    for n in pool:
        overall = overall_score.get(n, 0.0)
        best_pos_score = max(
            pos_scores.get(pos, {}).get(n, 0.0)
            for pos in range(5)
        )
        result[n] = (1 - w) * overall + w * best_pos_score

    return result


# ================ 辅助信号计算 ================

def _normalize_dict(scores):
    """归一化评分到 [0, 1]。"""
    vals = list(scores.values())
    if not vals:
        return scores
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {n: 0.5 for n in scores}
    return {n: (v - mn) / (mx - mn) for n, v in scores.items()}


def _weighted_blend(signals, weights, pool):
    """多信号加权融合。"""
    result = {}
    for n in pool:
        total = 0.0
        w_sum = 0.0
        for name, score_dict in signals.items():
            w = weights.get(name, 0.0)
            s = score_dict.get(n, 0.0)
            total += w * s
            w_sum += w
        result[n] = total / w_sum if w_sum > 0 else 0.0
    return result


def _multi_window_score(stats, gs):
    """多窗口特征融合评分。"""
    ewma = {n: s.ewma_freq for n, s in stats.items()}
    bayes = {n: s.bayes_freq for n, s in stats.items()}
    omp = {n: s.omission_pct for n, s in stats.items()}
    mom = {n: max(0, s.momentum) for n, s in stats.items()}

    ne = _normalize_dict(ewma)
    nb = _normalize_dict(bayes)
    no = _normalize_dict(omp)
    nm = _normalize_dict(mom)

    return {
        n: 0.25 * ne.get(n, 0) + 0.25 * nb.get(n, 0)
        + 0.25 * (1 - no.get(n, 0)) + 0.25 * nm.get(n, 0)
        for n in stats
    }


def _conditional_score(stats, cooccur, pool):
    """条件概率评分：基于号码共生率。"""
    if not cooccur:
        return {n: 0.5 for n in stats}

    trad = _score_map(stats)
    top_nums = sorted(trad, key=lambda n: -trad.get(n, 0))[:10]

    scores = {}
    for n in pool:
        if n not in cooccur:
            scores[n] = 0.5
            continue
        co_rates = [cooccur[n].get(t, 0) for t in top_nums if t != n]
        scores[n] = sum(co_rates) / len(co_rates) if co_rates else 0.5

    return scores


def _conditional_score_digit(stats, gs, pos, records=None):
    """digit 条件概率评分：基于马尔可夫转移矩阵。"""
    pool = list(range(10))
    if records is None or len(records) < 20:
        return {d: 0.5 for d in pool}

    trans_count = [[0] * 10 for _ in range(10)]
    for idx in range(len(records) - 1, 0, -1):
        prev_digits = records[idx].get("front", [])
        curr_digits = records[idx - 1].get("front", [])
        if pos < len(prev_digits) and pos < len(curr_digits):
            from_d = prev_digits[pos]
            to_d = curr_digits[pos]
            if 0 <= from_d <= 9 and 0 <= to_d <= 9:
                trans_count[from_d][to_d] += 1

    latest = records[0].get("front", [])
    if pos >= len(latest):
        return {d: 0.5 for d in pool}

    last_digit = latest[pos]
    if not (0 <= last_digit <= 9):
        return {d: 0.5 for d in pool}

    row_total = sum(trans_count[last_digit])
    if row_total == 0:
        return {d: 0.5 for d in pool}

    return {d: trans_count[last_digit][d] / row_total for d in pool}


def _arima_score_lotto(gs, records=None, pool=None, zone="front"):
    """ARIMA 评分（简化为 AR(1) 预测）。"""
    if pool is None:
        pool = GAMES[gs.game]["front_pool"]
    if records is None or len(records) < 20:
        return {n: 0.5 for n in pool}

    scores = {}
    for n in pool:
        series = []
        for r in records:
            if zone == "front":
                series.append(1 if n in r.get("front", []) else 0)
            else:
                series.append(1 if n in r.get("back", []) else 0)
        scores[n] = _ar1_predict(series)

    return scores


def _arima_score_digit(gs, records=None, pos=0):
    """digit ARIMA 评分。"""
    pool = list(range(10))
    if records is None or len(records) < 20:
        return {d: 0.5 for d in pool}

    result = {}
    for d in pool:
        series = []
        for r in records:
            digits = r.get("front", [])
            series.append(1 if pos < len(digits) and digits[pos] == d else 0)
        result[d] = _ar1_predict(series)

    return result


def _ar1_predict(series):
    """AR(1) 模型预测：X_next = rho * X_latest + (1-rho) * mu。"""
    if len(series) < 5:
        return 0.5

    import numpy as np
    arr = np.array(series, dtype=float)
    mu = float(np.mean(arr))
    var_x = float(np.var(arr))
    if var_x < 1e-9:
        return mu

    x_curr = arr[:-1]
    x_prev = arr[1:]
    cov = float(np.mean((x_curr - mu) * (x_prev - mu)))
    rho = max(-0.9, min(0.9, cov / var_x))

    latest = arr[0]
    prediction = rho * latest + (1 - rho) * mu
    return float(max(0.0, min(1.0, prediction)))


# ================ 反冷号策略 ================

def _anti_cold_filter(stats, pool):
    """反冷号过滤：排除遗漏比过高的号码，只保留活跃号。"""
    active = [n for n in pool
              if stats.get(n) and stats[n].omission_ratio <= ANTI_COLD_MAX_OMISSION_RATIO]

    threshold = ANTI_COLD_MAX_OMISSION_RATIO
    while len(active) < max(5, len(pool) // 3) and threshold < 10:
        threshold += 0.5
        active = [n for n in pool
                  if stats.get(n) and stats[n].omission_ratio <= threshold]

    if not active:
        active = list(pool)

    return active


# ================ 软多样性惩罚 ================

def _soft_diversity_penalty(score, picked_counter):
    """软多样性惩罚（v4 增强：50%基础+75%重复）。"""
    result = {}
    for n, s in score.items():
        count = picked_counter.get(n, 0)
        if count >= DIVERSITY_HARD_EXCLUDE_THRESHOLD:
            result[n] = 0.0
        elif count == 2:
            result[n] = s * (1 - DIVERSITY_REPEAT_PENALTY)
        elif count == 1:
            result[n] = s * (1 - DIVERSITY_BASE_PENALTY)
        else:
            result[n] = s
    return result


# ================ 选号策略 ================

def _pick_by_score(stats, score, k, strategy, exclude=None):
    """按策略选 k 个号。不同策略产生真正不同的排序。"""
    exclude = exclude or set()
    pool = {n: s for n, s in stats.items() if n not in exclude}

    if not pool:
        pool = dict(stats)

    # 预计算频率排名（归一化到 0~1），用于反冷号策略
    freq_vals = {n: s.freq for n, s in pool.items()}
    max_freq = max(freq_vals.values()) if freq_vals else 1
    min_freq = min(freq_vals.values()) if freq_vals else 0
    freq_range = max_freq - min_freq if max_freq != min_freq else 1
    freq_rank = {n: (s.freq - min_freq) / freq_range for n, s in pool.items()}

    if strategy == "hot":
        ranked = sorted(pool, key=lambda n: -(pool[n].freq * score.get(n, 0)))
    elif strategy == "anti_cold":
        # 反冷号：优先选活跃但非最热的"温号"
        # 对最热号（freq_rank高）施加惩罚，对活跃温号施加奖励
        def anti_cold_key(n):
            s = score.get(n, 0)
            fr = freq_rank[n]
            # 温号奖励：freq_rank 在 0.3~0.7 区间获得加成
            warm_bonus = 1.0 + 0.3 * max(0, 1 - abs(fr - 0.5) / 0.2)
            # 最热号惩罚：freq_rank > 0.8 时降权
            hot_penalty = 0.6 if fr > 0.8 else 1.0
            return -(s * warm_bonus * hot_penalty)
        ranked = sorted(pool, key=anti_cold_key)
    elif strategy == "momentum":
        # 趋势加速：动量因子权重放大 2 倍
        ranked = sorted(pool, key=lambda n: -(pool[n].momentum ** 2 * score.get(n, 0)))
    elif strategy == "diverse":
        ranked = sorted(pool, key=lambda n: -score.get(n, 0))
        top_cut = max(k, int(len(ranked) * 0.6))
        top = ranked[:top_cut]
        random.shuffle(top)
        ranked = top + [n for n in ranked if n not in set(top)]
    elif strategy == "random":
        ranked = sorted(pool, key=lambda n: -score.get(n, 0))
        top = ranked[:max(k, len(ranked) // 2)]
        random.shuffle(top)
        ranked = top + [n for n in ranked if n not in set(top)]
    else:  # balanced
        ranked = sorted(pool, key=lambda n: -score.get(n, 0))
    return ranked[:k]


# ================ 真集成投票 ================

def _ensemble_vote_lotto(groups, front_pool, back_pool):
    """真集成投票：各组按权重投票。"""
    front_votes = Counter()
    back_votes = Counter()

    for idx, grp in enumerate(groups):
        picks = grp.get("picks", {})
        weight = ENSEMBLE_VOTE_WEIGHT_DECAY ** idx

        if "front" in picks:
            for n in picks["front"]:
                front_votes[n] += weight
        if "back" in picks:
            for n in picks["back"]:
                back_votes[n] += weight

    selected_front = [n for n, _ in front_votes.most_common(5)]
    selected_back = [n for n, _ in back_votes.most_common(2)]

    while len(selected_front) < 5:
        for n in sorted(front_pool):
            if n not in selected_front:
                selected_front.append(n)
                break
    while len(selected_back) < 2:
        for n in sorted(back_pool):
            if n not in selected_back:
                selected_back.append(n)
                break

    return {
        "picks": {"front": sorted(selected_front), "back": sorted(selected_back)},
        "front_votes": {str(n): round(front_votes.get(n, 0), 2) for n in selected_front},
        "back_votes": {str(n): round(back_votes.get(n, 0), 2) for n in selected_back},
        "label": "集成混合(加权投票)",
        "reason": (
            f"综合{len(groups)}组策略的加权投票结果。"
            f"前区得票：{', '.join(f'{n}({front_votes[n]:.1f}票)' for n in selected_front[:3])}。"
            f"集成投票综合多策略优势，比单一策略更稳健。"
        ),
    }


def _ensemble_vote_digit(groups, positions):
    """digit 真集成投票。"""
    pos_counters = [Counter() for _ in range(positions)]

    for idx, grp in enumerate(groups):
        digits = grp.get("picks", {}).get("digits", [])
        weight = ENSEMBLE_VOTE_WEIGHT_DECAY ** idx
        for pos in range(min(len(digits), positions)):
            pos_counters[pos][digits[pos]] += weight

    selected = []
    for pos in range(positions):
        best_digit = pos_counters[pos].most_common(1)[0][0] if pos_counters[pos] else 0
        selected.append(best_digit)

    return {
        "picks": {"digits": selected},
        "votes": {str(pos): {str(d): round(c, 2) for d, c in pos_counters[pos].most_common(3)}
                  for pos in range(positions)},
        "label": "集成混合(加权投票)",
        "reason": (
            f"综合{len(groups)}组策略的加权投票结果。"
            + " ".join(f"第{pos+1}位={selected[pos]}({pos_counters[pos].get(selected[pos], 0):.1f}票)"
                       for pos in range(positions))
        ),
    }


# ================ 纯随机参考组 ================

def _lotto_random_group(gs, idx):
    """大乐透纯随机组。"""
    from config import GAMES
    cfg = GAMES[gs.game]
    front = sorted(random.sample(cfg["front_pool"], cfg["front_pick"]))
    back = sorted(random.sample(cfg["back_pool"], cfg["back_pick"]))
    picks = {"front": front, "back": back}
    reason = (
        "\U0001f3b2 纯随机抽样。回测消融实验表明：任何基于频率/遗漏/趋势的特征工程，"
        "其命中率与纯随机无统计显著差异。"
    )
    return {"index": idx, "label": "纯随机(诚实基线)", "picks": picks,
            "reason": reason, "scores": {}, "bets": 1,
            "cost": BET_UNIT_PRICE.get("dlt", 2)}


def _digit_random_group(gs, idx, game="pl3"):
    """digit 纯随机组。"""
    from config import GAMES
    cfg = GAMES[game]
    positions = cfg["positions"]
    picks = [random.randint(0, 9) for _ in range(positions)]
    reason = (
        "\U0001f3b2 纯随机抽样。回测显示排列3/5的任何特征工程命中率约等于1/10（随机基线）。"
    )
    return {"index": idx, "label": "纯随机(诚实基线)",
            "picks": {"digits": picks}, "reason": reason,
            "scores": {}, "bets": 1, "cost": BET_UNIT_PRICE.get(game, 2)}


# ================ 大乐透推荐 ================

def _lotto_single_group(gs, front_score, back_score,
                        strategy, label, idx,
                        front_exclude=None, back_exclude=None):
    """生成一组大乐透单选。"""
    front = _pick_by_score(gs.front_stats, front_score, 5, strategy, exclude=front_exclude)
    back = _pick_by_score(gs.back_stats, back_score, 2, strategy, exclude=back_exclude)
    front.sort()
    back.sort()
    picks = {"front": front, "back": back}
    score_snap = {
        "front": {str(n): round(front_score.get(n, 0), 4) for n in front},
        "back": {str(n): round(back_score.get(n, 0), 4) for n in back},
    }
    reason = explain_group(gs, picks, label, strategy)
    return {"index": idx, "label": label, "picks": picks,
            "reason": reason, "scores": score_snap, "bets": 1,
            "cost": BET_UNIT_PRICE.get("dlt", 2)}


def _lotto_combo(gs, front_score, back_score, mode):
    """生成一档复式。"""
    kf, kb = COMBO_MODES["lotto"][mode]
    front = sorted(_pick_by_score(gs.front_stats, front_score, kf, "balanced"))
    back = sorted(_pick_by_score(gs.back_stats, back_score, kb, "balanced"))
    picks = {"front": front, "back": back}
    bets = len(front) * len(back) if mode != "single" else 1
    cost = bets * BET_UNIT_PRICE.get("dlt", 2)
    reason = explain_combo(gs, picks, mode, bets)
    score_snap = {
        "front": {str(n): round(front_score.get(n, 0), 4) for n in front},
        "back": {str(n): round(back_score.get(n, 0), 4) for n in back},
    }
    return {"index": 0, "label": f"{'单注' if mode=='single' else mode}",
            "picks": picks, "reason": reason, "scores": score_snap,
            "bets": bets, "cost": cost}


def _constrain_front(gs, front_score, exclude_front=None):
    """从高分号中挑选 5 个，使其和值/奇偶比/共生率/间距综合最优。"""
    candidates = sorted(gs.front_stats, key=lambda n: -front_score.get(n, 0))[:15]
    if exclude_front:
        candidates = [c for c in candidates if c not in exclude_front]
    if len(candidates) < 5:
        candidates = sorted(gs.front_stats, key=lambda n: -front_score.get(n, 0))[:15 + len(exclude_front or set())]
    lo_sum, hi_sum = gs.sum_range
    target_odd = int(gs.odd_ratio_common.split(":")[0]) if gs.odd_ratio_common else 3
    from itertools import combinations
    all_combos = list(combinations(candidates, 5))

    raw_scores = [sum(front_score.get(n, 0) for n in combo) for combo in all_combos]
    raw_min = min(raw_scores) if raw_scores else 0
    raw_max = max(raw_scores) if raw_scores else 1
    raw_range = raw_max - raw_min if raw_max > raw_min else 1

    cooc = gs.cooccur
    co_avgs = []
    for combo in all_combos:
        rates = []
        for i, a in enumerate(combo):
            for b in combo[i + 1:]:
                rates.append(cooc.get(a, {}).get(b, 0))
        co_avgs.append(sum(rates) / len(rates) if rates else 0)
    co_min = min(co_avgs) if co_avgs else 0
    co_max = max(co_avgs) if co_avgs else 1
    co_range = co_max - co_min if co_max > co_min else 1

    best, best_score = None, -1
    for idx, (combo, raw) in enumerate(zip(all_combos, raw_scores)):
        normed = (raw - raw_min) / raw_range * 2
        sc = normed
        s = sum(combo)
        odd = sum(1 for x in combo if x % 2 == 1)
        if lo_sum <= s <= hi_sum:
            sc += 2
        if odd == target_odd:
            sc += 1
        co_normed = (co_avgs[idx] - co_min) / co_range * 1.5
        sc += co_normed
        if gs.gap_dist:
            from .features import gap_score
            sc += gap_score(list(combo), gs.gap_dist)
        if sc > best_score:
            best_score, best = sc, list(combo)
    return sorted(best) if best else sorted(candidates[:5])


def _recommend_lotto(gs, records=None):
    """大乐透推荐主函数（v4 多信号融合版）。"""
    front_score, back_score = _fused_score_lotto(gs, records)

    front_counter = Counter()
    back_counter = Counter()
    groups = []

    # 组1: 均衡热号
    g1 = _lotto_single_group(gs, front_score, back_score, "balanced", "均衡热号", 1)
    groups.append(g1)
    front_counter.update(g1["picks"]["front"])
    back_counter.update(g1["picks"]["back"])

    # 组2: 反冷号+条件概率
    active_front_score = front_score.copy()
    active_pool = _anti_cold_filter(gs.front_stats, GAMES[gs.game]["front_pool"])
    for n in front_score:
        if n not in active_pool:
            active_front_score[n] *= 0.3

    penalized_front = _soft_diversity_penalty(active_front_score, front_counter)
    penalized_back = _soft_diversity_penalty(back_score, back_counter)

    g2 = _lotto_single_group(gs, penalized_front, penalized_back,
                              "anti_cold", "反冷号+条件概率", 2)
    groups.append(g2)
    front_counter.update(g2["picks"]["front"])
    back_counter.update(g2["picks"]["back"])

    # 组3: 趋势加速
    penalized_front = _soft_diversity_penalty(front_score, front_counter)
    penalized_back = _soft_diversity_penalty(back_score, back_counter)
    g3 = _lotto_single_group(gs, penalized_front, penalized_back,
                              "momentum", "趋势加速", 3)
    groups.append(g3)
    front_counter.update(g3["picks"]["front"])
    back_counter.update(g3["picks"]["back"])

    # 组4: 组合合规
    penalized_front = _soft_diversity_penalty(front_score, front_counter)
    penalized_back = _soft_diversity_penalty(back_score, back_counter)
    g4 = _lotto_single_group(gs, penalized_front, penalized_back,
                              "balanced", "组合合规", 4)
    g4["picks"]["front"] = _constrain_front(gs, front_score,
                                             exclude_front={n for n, c in front_counter.items() if c >= 2})
    g4["reason"] = explain_group(gs, g4["picks"], "组合合规", "balanced")
    groups.append(g4)
    front_counter.update(g4["picks"]["front"])
    back_counter.update(g4["picks"]["back"])

    # 组5: 幸运组合
    penalized_front = _soft_diversity_penalty(front_score, front_counter)
    penalized_back = _soft_diversity_penalty(back_score, back_counter)
    g5 = _lotto_single_group(gs, penalized_front, penalized_back,
                              "diverse", "幸运组合", 5)
    groups.append(g5)
    front_counter.update(g5["picks"]["front"])
    back_counter.update(g5["picks"]["back"])

    # 组6: 纯随机参考
    g6 = _lotto_random_group(gs, idx=6)
    groups.append(g6)

    # 组7: 集成混合(加权投票)
    blend = _ensemble_vote_lotto(
        [g for g in groups if g["label"] != "纯随机(诚实基线)"],
        list(gs.front_stats.keys()), list(gs.back_stats.keys()))
    blend["index"] = 7
    blend["bets"] = 1
    blend["cost"] = BET_UNIT_PRICE.get("dlt", 2)
    groups.append(blend)

    combos = {
        "small": _lotto_combo(gs, front_score, back_score, "small"),
        "medium": _lotto_combo(gs, front_score, back_score, "medium"),
    }
    return {"single": groups, "combos": combos}


# ================ 排列玩法推荐 ================

def _constrain_digits(gs, pos_scores, pos_exclude=None):
    """digit 组合合规组。"""
    positions = len(gs.position_stats)
    top_k = DIGIT_COMBO_TOP_K
    candidates_per_pos = []
    for pos in range(positions):
        stats = gs.position_stats[pos]
        score = pos_scores.get(pos, _score_map(stats))
        ranked = sorted(stats, key=lambda n: -score.get(n, 0))
        if pos_exclude and pos in pos_exclude:
            ranked = [d for d in ranked if d not in pos_exclude[pos]]
        if len(ranked) < top_k:
            all_ranked = sorted(stats, key=lambda n: -score.get(n, 0))
            ranked = all_ranked[:max(top_k, 5)]
        candidates_per_pos.append(ranked[:top_k])

    from itertools import product
    lo_sum, hi_sum = gs.digit_sum_range
    target_odd = gs.digit_odd_count_common

    all_combos = list(product(*candidates_per_pos))
    if not all_combos:
        return [candidates_per_pos[p][0] for p in range(positions)]

    raw_scores = []
    for combo in all_combos:
        s = sum(pos_scores.get(pos, _score_map(gs.position_stats[pos])).get(digit, 0)
                for pos, digit in enumerate(combo))
        raw_scores.append(s)
    raw_min = min(raw_scores)
    raw_max = max(raw_scores)
    raw_range = raw_max - raw_min if raw_max > raw_min else 1

    best, best_score = None, -1
    for combo, raw in zip(all_combos, raw_scores):
        normed = (raw - raw_min) / raw_range * 2
        sc = normed
        s = sum(combo)
        odd = sum(1 for d in combo if d % 2 == 1)
        if lo_sum <= s <= hi_sum:
            sc += 2
        if odd == target_odd:
            sc += 1
        if sc > best_score:
            best_score, best = sc, list(combo)
    return best if best else [candidates_per_pos[p][0] for p in range(positions)]


def _digit_single_group(gs, strategy, label, idx,
                        pos_scores=None, game="pl3",
                        pos_exclude=None):
    """生成一组 digit 单选。"""
    picks = []
    for pos in range(len(gs.position_stats)):
        stats = gs.position_stats[pos]
        score = pos_scores[pos] if pos_scores and pos in pos_scores else _score_map(stats)
        exclude_set = pos_exclude.get(pos, set()) if pos_exclude else set()
        chosen = _pick_by_score(stats, score, 1, strategy, exclude=exclude_set)
        picks.append(chosen[0])
    score_snap = {
        str(pos): {str(n): round(_score_map(gs.position_stats[pos]).get(n, 0), 4)
                   for n in [picks[pos]]}
        for pos in range(len(picks))
    }
    reason = explain_group(gs, {"digits": picks}, label, strategy)
    return {"index": idx, "label": label, "picks": {"digits": picks},
            "reason": reason, "scores": score_snap, "bets": 1,
            "cost": BET_UNIT_PRICE.get(game, 2)}


def _digit_combo(gs, mode, game="pl3"):
    """digit 复式。"""
    k = COMBO_MODES["digit"][mode]
    per_pos = {}
    digits_grid = []
    for pos in range(len(gs.position_stats)):
        stats = gs.position_stats[pos]
        score = _score_map(stats)
        chosen = _pick_by_score(stats, score, k, "balanced")
        per_pos[str(pos)] = {str(n): round(score.get(n, 0), 4) for n in chosen}
        digits_grid.append(chosen)
    bets = 1
    for g in digits_grid:
        bets *= len(g)
    cost = bets * BET_UNIT_PRICE.get(game, 2)
    picks = {"digits_grid": digits_grid}
    reason = explain_combo(gs, picks, mode, bets)
    return {"index": 0, "label": mode, "picks": picks,
            "reason": reason, "scores": per_pos, "bets": bets, "cost": cost}

# ================ 推荐入口 ================

def _recommend_digit(gs, game="pl3", records=None):
    """排列玩法推荐主函数（v4 多信号融合 + 组间多样性版）。"""
    pos_scores = _fused_score_digit(gs, records)
    pos_picked = {p: Counter() for p in range(len(gs.position_stats))}  # 组间多样性追踪
    groups = []

    # 组1: 均衡热号
    g1 = _digit_single_group(gs, "balanced", "均衡热号", 1,
                              pos_scores=pos_scores, game=game)
    groups.append(g1)
    for p, d in enumerate(g1["picks"]["digits"]):
        pos_picked[p][d] += 1

    # 组2: 反冷号+条件概率（对已选号施加多样性惩罚）
    active_pos_scores = {}
    for pos in range(len(gs.position_stats)):
        s = pos_scores.get(pos, {}).copy()
        stats = gs.position_stats[pos]
        active_pool = _anti_cold_filter(stats, list(range(10)))
        for n in s:
            if n not in active_pool:
                s[n] *= 0.3
        # 组间多样性惩罚
        s = _soft_diversity_penalty(s, pos_picked[pos])
        active_pos_scores[pos] = s

    g2 = _digit_single_group(gs, "anti_cold", "反冷号+条件概率", 2,
                              pos_scores=active_pos_scores, game=game)
    groups.append(g2)
    for p, d in enumerate(g2["picks"]["digits"]):
        pos_picked[p][d] += 1

    # 组3: 趋势加速（对已选号施加多样性惩罚）
    momentum_pos_scores = {}
    for pos in range(len(gs.position_stats)):
        s = _soft_diversity_penalty(pos_scores.get(pos, {}), pos_picked[pos])
        momentum_pos_scores[pos] = s

    g3 = _digit_single_group(gs, "momentum", "趋势加速", 3,
                              pos_scores=momentum_pos_scores, game=game)
    groups.append(g3)
    for p, d in enumerate(g3["picks"]["digits"]):
        pos_picked[p][d] += 1

    # 组4: 幸运组合（对已选号施加多样性惩罚）
    diverse_pos_scores = {}
    for pos in range(len(gs.position_stats)):
        s = _soft_diversity_penalty(pos_scores.get(pos, {}), pos_picked[pos])
        diverse_pos_scores[pos] = s

    g4 = _digit_single_group(gs, "diverse", "幸运组合", 4,
                              pos_scores=diverse_pos_scores, game=game)
    groups.append(g4)

    # 组5: 纯随机参考
    g5 = _digit_random_group(gs, idx=5, game=game)
    groups.append(g5)

    # 组6: 集成混合(加权投票)
    blend = _ensemble_vote_digit(
        [g for g in groups if g["label"] != "纯随机(诚实基线)"],
        len(gs.position_stats))
    blend["index"] = 6
    blend["bets"] = 1
    blend["cost"] = BET_UNIT_PRICE.get(game, 2)
    groups.append(blend)

    combos = {
        "small": _digit_combo(gs, "small", game),
        "medium": _digit_combo(gs, "medium", game),
    }
    return {"single": groups, "combos": combos}
def recommend(game, gs=None):
    """推荐入口函数。

    Parameters
    ----------
    game : str
        "dlt" / "pl3" / "pl5"
    gs : GameStats, optional
        若为 None 则自动调用 analyze(game) 生成。

    Returns
    -------
    dict
        {"single": [...groups], "combos": {...}}
    """
    from .analyzer import analyze
    from .models import fetch_draws

    if gs is None:
        gs = analyze(game)

    # 获取历史记录用于 ARIMA / 条件概率信号
    records = fetch_draws(game, limit=500)

    if GAMES[game]["type"] == "lotto":
        return _recommend_lotto(gs, records=records)
    else:
        return _recommend_digit(gs, game=game, records=records)


def recommend_and_save(game, gs=None):
    """推荐并保存到数据库。"""
    from .models import save_recommendation
    from .analyzer import analyze

    if gs is None:
        gs = analyze(game)

    result = recommend(game, gs)

    # 保存单选组
    for g in result.get("single", []):
        save_recommendation(
            game=game,
            group_index=g["index"],
            mode="single",
            label=g["label"],
            picks=g["picks"],
            reason=g.get("reason", ""),
            scores=g.get("scores", {}),
            bets=g.get("bets", 1),
            cost=g.get("cost", 2),
        )

    # 保存复式
    for mode, c in result.get("combos", {}).items():
        save_recommendation(
            game=game,
            group_index=0,
            mode=mode,
            label=c["label"],
            picks=c["picks"],
            reason=c.get("reason", ""),
            scores=c.get("scores", {}),
            bets=c.get("bets", 1),
            cost=c.get("cost", 2),
        )

    return result


# ================ 集成融合辅助（供 detailed_backtest 调用） ================

def ensemble_blend_lotto(groups, front_pool, back_pool):
    """集成融合（大乐透）— 供外部模块调用。"""
    return _ensemble_vote_lotto(groups, front_pool, back_pool)


def ensemble_blend_digit(groups, positions):
    """集成融合（排列玩法）— 供外部模块调用。"""
    return _ensemble_vote_digit(groups, positions)
