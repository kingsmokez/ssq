# -*- coding: utf-8 -*-
"""精细化全管线回测 — 对每个测试期运行完整推荐引擎，记录所有方法×所有策略的命中细节。

v3 设计要点：
  - 每期运行真实 recommend() + ML-GBM + Stacking + ARIMA
  - 分别统计前区/后区/全中/每位
  - 大样本量（尽可能多的测试点）
  - 输出完整命中率矩阵
"""

import random as rnd
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from config import GAMES
from . import models
from .analyzer import _per_number_stats, _build_window, analyze_from_records
from .recommender import _score_map, recommend, ensemble_blend_lotto, ensemble_blend_digit
from .ml_scorer import MLScorer
from .arima_predictor import arima_scores_lotto, arima_scores_digit
from .stacking import StackingEnsemble


@dataclass
class PerPointResult:
    """单个测试点的回测结果。"""
    issue: str
    date: str
    # 实际开奖
    actual_front: List[int] = field(default_factory=list)
    actual_back: List[int] = field(default_factory=list)
    # 各方法命中详情 {method_name: {metric: value}}
    method_results: Dict[str, dict] = field(default_factory=dict)


@dataclass
class DetailedBacktestResult:
    """精细化回测的完整结果。"""
    game: str
    name: str
    n_points: int = 0
    # 各方法汇总 {method_name: {metric: value}}
    method_summary: Dict[str, dict] = field(default_factory=dict)
    # 各策略组汇总 {group_label: {metric: value}}
    group_summary: Dict[str, dict] = field(default_factory=dict)
    # 逐期详情
    per_point: List[PerPointResult] = field(default_factory=list)
    verdict: str = ""


def detailed_backtest(game: str, n_test: int = 200, step: int = 1) -> DetailedBacktestResult:
    """精细化全管线回测。

    n_test: 测试最近多少期
    step: 采样间隔（1=每期都测，2=隔1期测一次）
    """
    cfg = GAMES[game]
    is_lotto = cfg["type"] == "lotto"
    result = DetailedBacktestResult(game=game, name=cfg["name"])

    all_records = models.fetch_draws(game, limit=5000, order_desc=True)
    max_available = len(all_records) - 50
    if max_available < 10:
        return result

    actual_n = min(n_test, max_available)
    actual_step = max(1, step)

    # 初始化方法统计容器
    method_hits = defaultdict(int)   # {method: hits}
    method_picks = defaultdict(int)  # {method: picks}
    method_full = defaultdict(int)   # {method: full_hits}

    # 各策略组统计
    group_hits = defaultdict(int)
    group_picks = defaultdict(int)

    # 后区统计(lotto)
    back_method_hits = defaultdict(int)
    back_method_picks = defaultdict(int)

    # 每位统计(digit)
    pos_method_hits = defaultdict(lambda: defaultdict(int))
    pos_method_picks = defaultdict(lambda: defaultdict(int))

    n_points = 0

    for i in range(1, actual_n, actual_step):
        test_rec = all_records[i]
        train_records = all_records[i + 1: i + 800]
        if len(train_records) < 30:
            continue

        train_window = _build_window(train_records, 730)
        if len(train_window) < 20:
            train_window = train_records

        point = PerPointResult(
            issue=test_rec.get("issue", ""),
            date=test_rec.get("draw_date", "")[:10],
            actual_front=list(test_rec.get("front", [])),
            actual_back=list(test_rec.get("back", [])),
        )

        # ====== 运行所有评分方法 ======
        if is_lotto:
            actual_front_set = set(test_rec["front"])
            actual_back_set = set(test_rec.get("back", []))
            front_pool = cfg["front_pool"]
            back_pool = cfg["back_pool"]

            # 构建统计
            front_sets = [set(r["front"]) for r in train_window]
            back_sets = [set(r["back"]) for r in train_window]
            front_stats = _per_number_stats(front_pool, front_sets, len(train_window), 30, 50)
            back_stats = _per_number_stats(back_pool, back_sets, len(train_window), 30, 50)

            # 方法1: 传统评分 top-7
            trad_score = _score_map(front_stats)
            trad_ranked = sorted(trad_score, key=lambda n: -trad_score.get(n, 0))[:7]
            trad_hits_n = len(set(trad_ranked) & actual_front_set)
            method_hits["trad_top7"] += trad_hits_n
            method_picks["trad_top7"] += 7

            # 方法2: 传统评分 top-5 (真实单组大小)
            trad5 = sorted(trad_score, key=lambda n: -trad_score.get(n, 0))[:5]
            trad5_hits = len(set(trad5) & actual_front_set)
            method_hits["trad_top5"] += trad5_hits
            method_picks["trad_top5"] += 5

            # 方法3: ML-GBM top-7
            ml = MLScorer(game, model_type="gbm")
            ml.fit(train_records)
            ml_score = ml.predict(front_stats, front_pool, len(train_window))
            ml_ranked = sorted(ml_score, key=lambda n: -ml_score.get(n, 0))[:7]
            ml_hits_n = len(set(ml_ranked) & actual_front_set)
            method_hits["ml_gbm_top7"] += ml_hits_n
            method_picks["ml_gbm_top7"] += 7

            # 方法4: ML-GBM top-5
            ml5 = sorted(ml_score, key=lambda n: -ml_score.get(n, 0))[:5]
            ml5_hits = len(set(ml5) & actual_front_set)
            method_hits["ml_gbm_top5"] += ml5_hits
            method_picks["ml_gbm_top5"] += 5

            # 方法5: Stacking top-7
            ens = StackingEnsemble(game)
            front_fused, _ = ens.score_lotto(train_records)
            stack_ranked = sorted(front_fused, key=lambda n: -front_fused.get(n, 0))[:7]
            stack_hits_n = len(set(stack_ranked) & actual_front_set)
            method_hits["stacking_top7"] += stack_hits_n
            method_picks["stacking_top7"] += 7

            # 方法6: ARIMA top-7
            arima_score = arima_scores_lotto(train_window, front_pool)
            arima_ranked = sorted(arima_score, key=lambda n: -arima_score.get(n, 0))[:7]
            arima_hits_n = len(set(arima_ranked) & actual_front_set)
            method_hits["arima_top7"] += arima_hits_n
            method_picks["arima_top7"] += 7

            # 方法7: 随机基线 top-7
            rand7 = rnd.sample(front_pool, 7)
            rand_hits_n = len(set(rand7) & actual_front_set)
            method_hits["random_top7"] += rand_hits_n
            method_picks["random_top7"] += 7

            # 方法8: 随机基线 top-5
            rand5 = rnd.sample(front_pool, 5)
            rand5_hits = len(set(rand5) & actual_front_set)
            method_hits["random_top5"] += rand5_hits
            method_picks["random_top5"] += 5

            # ====== 全管线推荐 (6组+复式) ======
            gs = analyze_from_records(game, train_records)
            rec = recommend(game, gs)

            for grp in rec["single"]:
                label = grp["label"]
                front_picks = grp["picks"]["front"]
                back_picks = grp["picks"]["back"]
                fh = len(set(front_picks) & actual_front_set)
                bh = len(set(back_picks) & actual_back_set)
                group_hits[label] += fh
                group_picks[label] += len(front_picks)
                # 后区
                back_method_hits["group_" + label] += bh
                back_method_picks["group_" + label] += len(back_picks)
                # 全中
                if fh == len(front_picks) and bh == len(back_picks):
                    method_full["group_" + label] += 1

            # 集成混合
            non_random = [g for g in rec["single"] if g["label"] != "纯随机(诚实基线)"]
            blend = ensemble_blend_lotto(non_random, front_pool, back_pool)
            blend_front = blend["picks"]["front"]
            blend_fh = len(set(blend_front) & actual_front_set)
            group_hits["集成混合(加权投票)"] += blend_fh
            group_picks["集成混合(加权投票)"] += len(blend_front)

            # 复式
            for mode in ("small", "medium"):
                c = rec["combos"][mode]
                cfront = c["picks"]["front"]
                cback = c["picks"]["back"]
                ch = len(set(cfront) & actual_front_set)
                group_hits["复式_" + mode] += ch
                group_picks["复式_" + mode] += len(cfront)
                back_method_hits["combo_" + mode] += len(set(cback) & actual_back_set)
                back_method_picks["combo_" + mode] += len(cback)

            # ML推荐组 (用ML评分替代传统评分，5个号)
            ml5_group = sorted(ml_score, key=lambda n: -ml_score.get(n, 0))[:5]
            ml5g_fh = len(set(ml5_group) & actual_front_set)
            group_hits["ML-GBM推荐"] += ml5g_fh
            group_picks["ML-GBM推荐"] += 5

            # Stacking推荐组
            stack5 = sorted(front_fused, key=lambda n: -front_fused.get(n, 0))[:5]
            stack5_fh = len(set(stack5) & actual_front_set)
            group_hits["Stacking推荐"] += stack5_fh
            group_picks["Stacking推荐"] += 5

            # 后区独立统计
            # 传统后区评分 top-3
            back_trad = _score_map(back_stats)
            bt3 = sorted(back_trad, key=lambda n: -back_trad.get(n, 0))[:3]
            bt3_bh = len(set(bt3) & actual_back_set)
            back_method_hits["trad_back_top3"] += bt3_bh
            back_method_picks["trad_back_top3"] += 3

            # ML后区
            ml_back = ml.predict(back_stats, back_pool, len(train_window))
            mb3 = sorted(ml_back, key=lambda n: -ml_back.get(n, 0))[:3]
            mb3_bh = len(set(mb3) & actual_back_set)
            back_method_hits["ml_back_top3"] += mb3_bh
            back_method_picks["ml_back_top3"] += 3

            point.method_results = {
                "trad_top7": {"hits": trad_hits_n},
                "ml_gbm_top7": {"hits": ml_hits_n},
                "stacking_top7": {"hits": stack_hits_n},
                "arima_top7": {"hits": arima_hits_n},
                "random_top7": {"hits": rand_hits_n},
            }

        else:
            # ====== Digit 玩法 ======
            positions = cfg["positions"]
            actual = list(test_rec["front"])
            pool = cfg["digit_pool"]

            for pos in range(positions):
                seq = []
                for r in train_window:
                    digits = r["front"]
                    seq.append({digits[pos]} if pos < len(digits) else set())
                stats = _per_number_stats(pool, seq, len(train_window), 30, 50)

                # 传统评分
                trad_score = _score_map(stats)
                trad_best = max(trad_score, key=trad_score.get)
                trad_hit = 1 if trad_best == actual[pos] else 0
                method_hits["trad_per_pos"] += trad_hit
                method_picks["trad_per_pos"] += 1
                pos_method_hits[pos]["trad"] += trad_hit
                pos_method_picks[pos]["trad"] += 1

                # ARIMA
                arima_all = arima_scores_digit(train_window, positions)
                arima_best = max(arima_all[pos], key=arima_all[pos].get)
                arima_hit = 1 if arima_best == actual[pos] else 0
                method_hits["arima_per_pos"] += arima_hit
                method_picks["arima_per_pos"] += 1
                pos_method_hits[pos]["arima"] += arima_hit
                pos_method_picks[pos]["arima"] += 1

                # 随机
                rand_d = rnd.randint(0, 9)
                rand_hit = 1 if rand_d == actual[pos] else 0
                method_hits["random_per_pos"] += rand_hit
                method_picks["random_per_pos"] += 1
                pos_method_hits[pos]["random"] += rand_hit
                pos_method_picks[pos]["random"] += 1

            # 全管线推荐
            gs = analyze_from_records(game, train_records)
            rec = recommend(game, gs)
            for grp in rec["single"]:
                label = grp["label"]
                digits = grp["picks"]["digits"]
                hits = sum(1 for j in range(min(len(digits), len(actual)))
                          if digits[j] == actual[j])
                group_hits[label] += hits
                group_picks[label] += len(digits)

            # 复式覆盖
            for mode in ("small", "medium"):
                c = rec["combos"][mode]
                grid = c["picks"]["digits_grid"]
                pos_hits = 0
                for pos in range(min(len(grid), len(actual))):
                    if actual[pos] in grid[pos]:
                        pos_hits += 1
                group_hits["复式_" + mode] += pos_hits
                group_picks["复式_" + mode] += len(grid)

        result.per_point.append(point)
        n_points += 1

    result.n_points = n_points

    # ====== 汇总 ======
    # 方法汇总
    for m in method_hits:
        picks = method_picks.get(m, 1)
        hits = method_hits[m]
        result.method_summary[m] = {
            "hits": hits,
            "picks": picks,
            "hit_rate": round(hits / picks, 4) if picks > 0 else 0,
        }

    # 各策略组汇总
    for g in group_hits:
        picks = group_picks.get(g, 1)
        hits = group_hits[g]
        result.group_summary[g] = {
            "hits": hits,
            "picks": picks,
            "hit_rate": round(hits / picks, 4) if picks > 0 else 0,
        }

    # 后区汇总
    for m in back_method_hits:
        picks = back_method_picks.get(m, 1)
        hits = back_method_hits[m]
        result.method_summary[m] = {
            "hits": hits,
            "picks": picks,
            "hit_rate": round(hits / picks, 4) if picks > 0 else 0,
        }

    # 生成结论
    lines = []
    lines.append(f"共 {n_points} 个测试点。")

    # 找最佳方法
    best_method = max(result.method_summary.items(),
                     key=lambda x: x[1]["hit_rate"]) if result.method_summary else None
    if best_method:
        lines.append(
            f"最佳方法: {best_method[0]} 命中率 {best_method[1]['hit_rate']:.2%} "
            f"(命中{best_method[1]['hits']}/{best_method[1]['picks']})")

    # 找最佳策略组
    best_group = max(result.group_summary.items(),
                     key=lambda x: x[1]["hit_rate"]) if result.group_summary else None
    if best_group:
        lines.append(
            f"最佳策略: {best_group[0]} 命中率 {best_group[1]['hit_rate']:.2%} "
            f"(命中{best_group[1]['hits']}/{best_group[1]['picks']})")

    # 随机基线对比
    rand_key = "random_top7" if is_lotto else "random_per_pos"
    rand_rate = result.method_summary.get(rand_key, {}).get("hit_rate", 0)
    if best_method and rand_rate > 0:
        diff = best_method[1]["hit_rate"] - rand_rate
        lines.append(f"vs 随机基线 {rand_rate:.2%}: {'优于' if diff > 0 else '不如'}随机 ({diff:+.2%})")

    result.verdict = " ".join(lines)
    return result


def print_detailed_report(game: str, n_test: int = 200, step: int = 1):
    """运行精细化回测并打印报告。"""
    result = detailed_backtest(game, n_test=n_test, step=step)
    cfg = GAMES[game]
    is_lotto = cfg["type"] == "lotto"

    print(f"\n{'='*70}")
    print(f"  {result.name} ({game}) 精细化回测报告")
    print(f"  测试点: {result.n_points}")
    print(f"{'='*70}")

    # 方法排名
    print(f"\n--- 评分方法排名 ({'前区top-7/5' if is_lotto else '按位top-1'}) ---")
    sorted_methods = sorted(result.method_summary.items(),
                           key=lambda x: -x[1]["hit_rate"])
    for name, data in sorted_methods:
        if "back" in name:
            continue
        bar = "█" * int(data["hit_rate"] * 100)
        print(f"  {name:20s} {data['hit_rate']:6.2%}  ({data['hits']:3d}/{data['picks']:3d})  {bar}")

    # 后区排名 (lotto only)
    if is_lotto:
        back_methods = {k: v for k, v in result.method_summary.items() if "back" in k}
        if back_methods:
            print(f"\n--- 后区方法排名 ---")
            for name, data in sorted(back_methods.items(), key=lambda x: -x[1]["hit_rate"]):
                print(f"  {name:20s} {data['hit_rate']:6.2%}  ({data['hits']:3d}/{data['picks']:3d})")

    # 策略组排名
    print(f"\n--- 推荐策略组排名 ---")
    for name, data in sorted(result.group_summary.items(), key=lambda x: -x[1]["hit_rate"]):
        bar = "█" * int(data["hit_rate"] * 100)
        print(f"  {name:20s} {data['hit_rate']:6.2%}  ({data['hits']:3d}/{data['picks']:3d})  {bar}")

    # 结论
    print(f"\n--- 结论 ---")
    print(f"  {result.verdict}")

    return result
