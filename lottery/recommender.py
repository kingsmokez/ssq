# -*- coding: utf-8 -*-
"""推荐引擎 — 基于统计快照生成 5 组单选 + 小/中复式。

设计要点：
- 综合分 score = w_freq*freq + w_omission*norm(omission_ratio) + w_momentum*norm(momentum)
- 5 组单选各有明确"选号论点"，避免雷同：
  1. 均衡热号  2. 冷号回补  3. 趋势加速  4. 组合合规  5. 均衡随机
- 复式档位：lotto 前区/后区各取 Top-K；digit 每位取 Top-K。
"""

import random
from datetime import datetime
from typing import Dict, List

from config import WEIGHTS, GAMES, COMBO_MODES, DIGIT_COMBO_TOP_K, BET_UNIT_PRICE
from .analyzer import GameStats, NumberStat, analyze
from .explainer import explain_group, explain_combo


def _norm(values: Dict[int, float]) -> Dict[int, float]:
    """z-score 归一化，将值映射到 [0, 1]。

    使用 z-score (mean/std) 而非 min-max，避免单个极值压缩其他值的区分度。
    z ∈ [-3σ, +3σ] 映射到 [0, 1]，均值对应 0.5。
    """
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


def _score_map(stats: Dict[int, NumberStat]) -> Dict[int, float]:
    """对一组 NumberStat 计算综合分（v2 增强版）。

    v2 改进（基于回测反馈，避免"追热"反向）：
      - 用 ewma_freq 替代原始 freq：近期加权但更平滑，减少对瞬时热号的过度追逐
      - 用 bayes_freq 作为第二信号：抗小样本噪声
      - 用 omission_pct（遗漏分位）替代 omission_ratio：更抗极值
      - momentum 保留但权重降低（趋势信号噪声大）
      - 整体降低对"集中热号"的偏向，使评分更分散、更稳健

    ⚠️ 即便如此，回测显示这些特征仍不能真正预测彩票（独立随机事件），
       改进的价值是让推荐"合理分散"而非"反向"。
    """
    ewma = {n: getattr(s, "ewma_freq", s.freq) for n, s in stats.items()}
    bayes = {n: getattr(s, "bayes_freq", s.freq) for n, s in stats.items()}
    omp = {n: getattr(s, "omission_pct", 0.5) for n, s in stats.items()}
    mom = {n: s.momentum for n, s in stats.items()}
    ne, nb, no, nm = _norm(ewma), _norm(bayes), _norm(omp), _norm(mom)
    # 频率类信号取 ewma 与 bayes 的均值，更稳健
    w = WEIGHTS
    return {
        n: (w["freq"] * 0.6 * ne.get(n, 0) + w["freq"] * 0.4 * nb.get(n, 0))
        + w["omission"] * no.get(n, 0)
        + w["momentum"] * 0.6 * nm.get(n, 0)   # 趋势噪声大，降权
        for n in stats
    }


def _pick_by_score(stats: Dict[int, NumberStat], score: Dict[int, float],
                   k: int, strategy: str, exclude=None) -> List[int]:
    """按策略选 k 个号。strategy: hot/cold/momentum/balanced/diverse/random。
    exclude: 已选号集合（lotto 前区去重）。"""
    exclude = exclude or set()
    pool = {n: s for n, s in stats.items() if n not in exclude}
    if strategy == "hot":
        ranked = sorted(pool, key=lambda n: -pool[n].freq)
    elif strategy == "cold":
        ranked = sorted(pool, key=lambda n: -pool[n].omission_ratio)
    elif strategy == "momentum":
        ranked = sorted(pool, key=lambda n: -pool[n].momentum)
    elif strategy == "diverse":
        # 多样性策略：优先选择未在前组出现的号码，再补充高分已选号
        ranked = sorted(pool, key=lambda n: -score.get(n, 0))
        # 在 Top 60% 范围内随机，增加多样性
        top_cut = max(k, int(len(ranked) * 0.6))
        top = ranked[:top_cut]
        random.shuffle(top)
        ranked = top + [n for n in ranked if n not in set(top)]
    elif strategy == "random":
        ranked = sorted(pool, key=lambda n: -score.get(n, 0))
        # 在 Top 1/2 范围内随机
        top = ranked[: max(k, len(ranked) // 2)]
        random.shuffle(top)
        ranked = top + [n for n in ranked if n not in set(top)]
    else:  # balanced: 综合分
        ranked = sorted(pool, key=lambda n: -score.get(n, 0))
    return ranked[:k]


def _diversify_score(score: Dict[int, float], already_picked: set,
                     penalty: float = 0.15) -> Dict[int, float]:
    """对已选号码施加惩罚，增加组间多样性。

    penalty: 已选号码的降权比例（0.15 = 降权15%）。
    """
    return {n: (s * (1 - penalty) if n in already_picked else s)
            for n, s in score.items()}


# ---------------- 纯随机参考组 ----------------
# 消融实验结论：纯随机选号的命中率（lift）≥ 任何特征工程配置。
# 这是彩票"独立随机事件"本质的直接体现——任何基于历史的信号都是噪声。
# 这一组作为"诚实基线"展示，让用户理解：热号/冷号/趋势并不比随机更准。

def _lotto_random_group(gs: GameStats, idx: int):
    """大乐透纯随机组：前区5个+后区2个，全部从号池随机抽样（不放回）。"""
    from config import GAMES
    cfg = GAMES[gs.game]
    front = sorted(random.sample(cfg["front_pool"], cfg["front_pick"]))
    back = sorted(random.sample(cfg["back_pool"], cfg["back_pick"]))
    picks = {"front": front, "back": back}
    reason = (
        "🎲 纯随机抽样。回测消融实验表明：任何基于频率/遗漏/趋势的特征工程，"
        "其命中率与纯随机无统计显著差异（lift 落在 ±10% 噪声带内）。"
        "这组的存在是为了诚实提醒——所谓'热号冷号'并不比随机更准。"
    )
    score_snap = {
        "front": {str(n): 0.0 for n in front},
        "back": {str(n): 0.0 for n in back},
    }
    return {"index": idx, "label": "纯随机(诚实基线)", "picks": picks,
            "reason": reason, "scores": score_snap, "bets": 1,
            "cost": BET_UNIT_PRICE.get("dlt", 2)}


def _digit_random_group(gs: GameStats, idx: int, game: str = "pl3"):
    """digit 纯随机组：每位独立从 0-9 随机。"""
    from config import GAMES
    cfg = GAMES[game]
    positions = cfg["positions"]
    picks = [random.randint(0, 9) for _ in range(positions)]
    reason = (
        "🎲 纯随机抽样。回测显示排列3/5的任何特征工程命中率 ≈ 1/10（随机基线），"
        "p值远大于0.05，统计上等同于掷骰子。这组提醒：别为'数字没中'自责，"
        "那是72.9%概率会发生的事。"
    )
    score_snap = {str(pos): {str(d): 0.0 for d in [picks[pos]]}
                  for pos in range(positions)}
    return {"index": idx, "label": "纯随机(诚实基线)",
            "picks": {"digits": picks}, "reason": reason,
            "scores": score_snap, "bets": 1, "cost": BET_UNIT_PRICE.get(game, 2)}


# ---------------- lotto（大乐透） ----------------

def _lotto_single_group(gs: GameStats, front_score, back_score,
                        strategy: str, label: str, idx: int):
    """生成一组大乐透单选。"""
    front = _pick_by_score(gs.front_stats, front_score, 5, strategy)
    back = _pick_by_score(gs.back_stats, back_score, 2, strategy)
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


def _lotto_combo(gs: GameStats, front_score, back_score, mode: str):
    """生成一档复式（前区/后区各取 Top-K）。"""
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


def _recommend_lotto(gs: GameStats):
    front_score = _score_map(gs.front_stats)
    back_score = _score_map(gs.back_stats)

    # 逐组生成，每组选号前对已选号码降权，确保组间多样性
    front_picked: set = set()
    back_picked: set = set()
    groups = []

    # 组1: 均衡热号（综合分最高）
    fs1 = _diversify_score(front_score, front_picked)
    bs1 = _diversify_score(back_score, back_picked)
    g1 = _lotto_single_group(gs, fs1, bs1, "balanced", "均衡热号", 1)
    groups.append(g1)
    front_picked.update(g1["picks"]["front"])
    back_picked.update(g1["picks"]["back"])

    # 组2: 冷号回补
    fs2 = _diversify_score(front_score, front_picked)
    bs2 = _diversify_score(back_score, back_picked)
    g2 = _lotto_single_group(gs, fs2, bs2, "cold", "冷号回补", 2)
    groups.append(g2)
    front_picked.update(g2["picks"]["front"])
    back_picked.update(g2["picks"]["back"])

    # 组3: 趋势加速
    fs3 = _diversify_score(front_score, front_picked)
    bs3 = _diversify_score(back_score, back_picked)
    g3 = _lotto_single_group(gs, fs3, bs3, "momentum", "趋势加速", 3)
    groups.append(g3)
    front_picked.update(g3["picks"]["front"])
    back_picked.update(g3["picks"]["back"])

    # 组4: 组合合规（和值/奇偶/大小落历史高频区间）
    fs4 = _diversify_score(front_score, front_picked)
    bs4 = _diversify_score(back_score, back_picked)
    g4 = _lotto_single_group(gs, fs4, bs4, "balanced", "组合合规", 4)
    g4["picks"]["front"] = _constrain_front(gs, fs4)
    g4["reason"] = explain_group(gs, g4["picks"], "组合合规", "balanced")
    groups.append(g4)
    front_picked.update(g4["picks"]["front"])
    back_picked.update(g4["picks"]["back"])

    # 组5: 幸运组合（多样性策略，扩大随机范围）
    fs5 = _diversify_score(front_score, front_picked, penalty=0.25)
    bs5 = _diversify_score(back_score, back_picked, penalty=0.25)
    g5 = _lotto_single_group(gs, fs5, bs5, "diverse", "幸运组合", 5)
    groups.append(g5)

    # 组6: 纯随机参考（消融实验证明：随机命中率 ≥ 任何特征工程，
    # 因为彩票是独立随机事件，无信号可提取。这组作为"诚实基线"）
    g6 = _lotto_random_group(gs, idx=6)
    groups.append(g6)

    combos = {
        "small": _lotto_combo(gs, front_score, back_score, "small"),
        "medium": _lotto_combo(gs, front_score, back_score, "medium"),
    }
    return {"single": groups, "combos": combos}


def _constrain_front(gs: GameStats, front_score) -> List[int]:
    """从高分号中挑选 5 个，使其和值/奇偶比/共生率综合最优。

    评分机制：
      - 原始分数归一化到 [0, 2]
      - 和值合规 +2，奇偶合规 +1
      - 共生率加分：组合内 10 个号码对的历史共生率均值归一化到 [0, 1.5]
        偏好"历史上经常一起开出"的号码组合
    """
    candidates = sorted(gs.front_stats, key=lambda n: -front_score.get(n, 0))[:15]
    lo_sum, hi_sum = gs.sum_range
    target_odd = int(gs.odd_ratio_common.split(":")[0]) if gs.odd_ratio_common else 3
    from itertools import combinations
    all_combos = list(combinations(candidates, 5))
    # 原始分数范围
    raw_scores = [sum(front_score.get(n, 0) for n in combo) for combo in all_combos]
    raw_min = min(raw_scores) if raw_scores else 0
    raw_max = max(raw_scores) if raw_scores else 1
    raw_range = raw_max - raw_min if raw_max > raw_min else 1
    # 共生率范围（预计算所有组合的共生均值，用于归一化）
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
        # 归一化原始分数到 [0, 2]
        normed = (raw - raw_min) / raw_range * 2
        sc = normed
        s = sum(combo)
        odd = sum(1 for x in combo if x % 2 == 1)
        if lo_sum <= s <= hi_sum:
            sc += 2  # 和值合规
        if odd == target_odd:
            sc += 1  # 奇偶合规
        # 共生率加分：归一化到 [0, 1.5]
        co_normed = (co_avgs[idx] - co_min) / co_range * 1.5
        sc += co_normed
        if sc > best_score:
            best_score, best = sc, list(combo)
    return sorted(best) if best else sorted(candidates[:5])


# ---------------- digit（排列3/5） ----------------

def _constrain_digits(gs: GameStats, pos_scores: Dict[int, Dict[int, float]]) -> List[int]:
    """digit 组合合规组：从每位 Top-K 候选中枚举组合，选和值/奇偶最合规的。

    排列3 每位取 Top-3 → 3³=27 组合，排列5 每位取 Top-3 → 3⁵=243 组合，
    均可完整枚举。
    """
    positions = len(gs.position_stats)
    top_k = DIGIT_COMBO_TOP_K
    # 每位取 top-K 候选
    candidates_per_pos = []
    for pos in range(positions):
        stats = gs.position_stats[pos]
        score = pos_scores.get(pos, _score_map(stats))
        ranked = sorted(stats, key=lambda n: -score.get(n, 0))
        candidates_per_pos.append(ranked[:top_k])

    # 枚举所有组合
    from itertools import product
    lo_sum, hi_sum = gs.digit_sum_range
    target_odd = gs.digit_odd_count_common

    # 计算各组合的位置评分总和范围，用于归一化
    all_combos = list(product(*candidates_per_pos))
    if not all_combos:
        # 退化：每位取第1
        return [candidates_per_pos[p][0] for p in range(positions)]

    raw_scores = []
    for combo in all_combos:
        s = 0
        for pos, digit in enumerate(combo):
            s += pos_scores.get(pos, _score_map(gs.position_stats[pos])).get(digit, 0)
        raw_scores.append(s)
    raw_min = min(raw_scores)
    raw_max = max(raw_scores)
    raw_range = raw_max - raw_min if raw_max > raw_min else 1

    best, best_score = None, -1
    for combo, raw in zip(all_combos, raw_scores):
        # 归一化原始分数到 [0, 2]
        normed = (raw - raw_min) / raw_range * 2
        sc = normed
        s = sum(combo)
        odd = sum(1 for d in combo if d % 2 == 1)
        if lo_sum <= s <= hi_sum:
            sc += 2  # 和值合规
        if odd == target_odd:
            sc += 1  # 奇偶合规
        if sc > best_score:
            best_score, best = sc, list(combo)
    return best if best else [candidates_per_pos[p][0] for p in range(positions)]

def _digit_single_group(gs: GameStats, strategy: str, label: str, idx: int,
                        pos_scores: dict = None, game: str = "pl3"):
    """生成一组 digit 单选：每位取 Top-1。

    pos_scores: 可选的每位评分覆盖（用于多样性降权后传入）。
    """
    picks = []
    for pos in range(len(gs.position_stats)):
        stats = gs.position_stats[pos]
        score = pos_scores[pos] if pos_scores and pos in pos_scores else _score_map(stats)
        chosen = _pick_by_score(stats, score, 1, strategy)
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


def _digit_combo(gs: GameStats, mode: str, game: str = "pl3"):
    """digit 复式：每位取 Top-K。"""
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


def _recommend_digit(gs: GameStats, game: str = "pl3"):
    positions = len(gs.position_stats)
    # 计算每位的基础评分
    base_pos_scores = {pos: _score_map(gs.position_stats[pos])
                       for pos in range(positions)}

    # 逐组生成，每组选号前对已选数字降权
    digit_picked: Dict[int, set] = {pos: set() for pos in range(positions)}
    groups = []

    # 组1: 均衡热号
    ps1 = {pos: _diversify_score(base_pos_scores[pos], digit_picked[pos])
           for pos in range(positions)}
    g1 = _digit_single_group(gs, "balanced", "均衡热号", 1, pos_scores=ps1, game=game)
    groups.append(g1)
    for pos in range(positions):
        digit_picked[pos].add(g1["picks"]["digits"][pos])

    # 组2: 冷号回补
    ps2 = {pos: _diversify_score(base_pos_scores[pos], digit_picked[pos])
           for pos in range(positions)}
    g2 = _digit_single_group(gs, "cold", "冷号回补", 2, pos_scores=ps2, game=game)
    groups.append(g2)
    for pos in range(positions):
        digit_picked[pos].add(g2["picks"]["digits"][pos])

    # 组3: 趋势加速
    ps3 = {pos: _diversify_score(base_pos_scores[pos], digit_picked[pos])
           for pos in range(positions)}
    g3 = _digit_single_group(gs, "momentum", "趋势加速", 3, pos_scores=ps3, game=game)
    groups.append(g3)
    for pos in range(positions):
        digit_picked[pos].add(g3["picks"]["digits"][pos])

    # 组4: 组合合规（和值/奇偶落历史高频区间）
    g4 = _digit_single_group(gs, "balanced", "组合合规", 4, game=game)
    g4["picks"]["digits"] = _constrain_digits(gs, base_pos_scores)
    g4["reason"] = explain_group(gs, g4["picks"], "组合合规", "balanced")
    groups.append(g4)
    for pos in range(positions):
        digit_picked[pos].add(g4["picks"]["digits"][pos])

    # 组5: 幸运组合（多样性策略）
    ps5 = {pos: _diversify_score(base_pos_scores[pos], digit_picked[pos], penalty=0.25)
           for pos in range(positions)}
    g5 = _digit_single_group(gs, "diverse", "幸运组合", 5, pos_scores=ps5, game=game)
    groups.append(g5)

    # 组6: 纯随机参考（消融实验证明：digit 随机命中率 ≥ 任何特征工程）
    g6 = _digit_random_group(gs, idx=6, game=game)
    groups.append(g6)

    combos = {
        "small": _digit_combo(gs, "small", game),
        "medium": _digit_combo(gs, "medium", game),
    }
    return {"single": groups, "combos": combos}


# ---------------- 入口 ----------------

def recommend(game: str, gs: GameStats = None) -> dict:
    """生成某玩法的推荐（含 5 组单选 + 小/中复式）。gs 可传入避免重算。"""
    if gs is None:
        gs = analyze(game)
    if GAMES[game]["type"] == "lotto":
        return _recommend_lotto(gs)
    return _recommend_digit(gs, game)


def recommend_and_save(game: str, gs: GameStats = None) -> dict:
    """生成推荐并落库（用统一时间戳写新一批，确保同批次记录 created_at 一致）。"""
    from . import models
    result = recommend(game, gs)
    # 统一时间戳，确保同一批次所有记录的 created_at 一致
    batch_ts = datetime.now().isoformat(timespec="seconds")
    # 单选
    for g in result["single"]:
        models.save_recommendation(
            game, g["index"], "single", g["label"],
            g["picks"], g["reason"], g["scores"], created_at=batch_ts,
            bets=g.get("bets", 1), cost=g.get("cost", 2))
    # 复式
    for mode in ("small", "medium"):
        c = result["combos"][mode]
        models.save_recommendation(
            game, 0, mode, c["label"], c["picks"], c["reason"], c["scores"],
            created_at=batch_ts, bets=c.get("bets", 1), cost=c.get("cost", 2))
    return result


if __name__ == "__main__":
    from . import models
    models.init_db()
    for g in ("dlt", "pl3", "pl5"):
        gs = analyze(g)
        print(f"\n{'='*60}\n{gs.name} 推荐")
        res = recommend(g, gs)
        print("-- 单选 5 组 --")
        for grp in res["single"]:
            p = grp["picks"]
            if "front" in p:
                print(f"  [{grp['label']}] 前{p['front']} 后{p['back']}")
            else:
                print(f"  [{grp['label']}] {p['digits']}")
            print(f"     理由: {grp['reason'][:80]}...")
        print("-- 复式 --")
        for mode in ("small", "medium"):
            c = res["combos"][mode]
            p = c["picks"]
            if "front" in p:
                print(f"  {mode}({c['bets']}注): 前{p['front']} 后{p['back']}")
            else:
                print(f"  {mode}({c['bets']}注): {p['digits_grid']}")
