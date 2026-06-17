# -*- coding: utf-8 -*-
"""自适应权重衰减 — 根据近期命中反馈动态调整各评分器的权重。

核心思路：
  每次开奖后，检查每个评分器推荐的号码命中了多少，
  对命中多的评分器增加权重，命中少的降低权重。

  使用指数衰减（类似 EWMA）来平滑权重变化，
  避免单期噪声导致权重剧烈波动。
"""

import math
from typing import Dict, List
from collections import defaultdict

from config import GAMES
from . import models
from .analyzer import _per_number_stats, _build_window, analyze_from_records
from .recommender import _score_map, recommend
from .ml_scorer import MLScorer
from .arima_predictor import arima_scores_lotto, arima_scores_digit
from .stacking import StackingEnsemble


class AdaptiveEnsemble:
    """自适应集成 — 根据近期反馈动态调整评分器权重。"""

    def __init__(self, game: str, decay: float = 0.9):
        """初始化。

        game: 玩法代码
        decay: 权重衰减系数（0.9=慢适应，0.5=快适应）
        """
        self.game = game
        self.cfg = GAMES[game]
        self.decay = decay
        self.scorer_names = ["ml_gbm", "traditional", "arima"]

        # 初始权重（均匀）
        self.weights = {s: 1.0 / len(self.scorer_names) for s in self.scorer_names}
        # 累积命中/未命中
        self.hit_counts = {s: 0 for s in self.scorer_names}
        self.miss_counts = {s: 0 for s in self.scorer_names}
        # 近期表现（EWMA）
        self.recent_performance = {s: 0.5 for s in self.scorer_names}

    def update_weights(self, scores_dict: dict, actual_set: set, top_k: int = 7):
        """根据本轮命中结果更新各评分器权重。

        scores_dict: {scorer_name: {number: score}}
        actual_set: 实际开出的号码集合
        top_k: 每个评分器选 top-k 个号
        """
        for name, scores in scores_dict.items():
            if name not in self.weights:
                continue
            # 选 top-k
            ranked = sorted(scores, key=lambda n: -scores.get(n, 0))[:top_k]
            hits = len(set(ranked) & actual_set)
            hit_rate = hits / top_k if top_k > 0 else 0

            # 更新累积统计
            if hits > 0:
                self.hit_counts[name] += 1
            else:
                self.miss_counts[name] += 1

            # EWMA 更新近期表现
            self.recent_performance[name] = (
                self.decay * self.recent_performance[name] +
                (1 - self.decay) * hit_rate
            )

        # 根据近期表现重新分配权重
        total_perf = sum(max(0.01, self.recent_performance[s]) for s in self.scorer_names)
        if total_perf > 0:
            for s in self.scorer_names:
                self.weights[s] = max(0.05, self.recent_performance[s]) / total_perf

    def backtest_adaptive(self, n_test: int = 80) -> dict:
        """自适应集成回测。"""
        all_records = models.fetch_draws(self.game, limit=3000, order_desc=True)
        if len(all_records) < n_test + 50:
            n_test = max(5, len(all_records) - 50)
        if len(all_records) < 30:
            return {"error": "数据不足"}

        is_lotto = self.cfg["type"] == "lotto"
        adapt_hits = fixed_hits = 0
        adapt_picks = fixed_picks = 0
        n_points = 0
        weight_history = []

        step = max(1, n_test // 80)
        max_i = min(n_test, len(all_records) - 30)

        for i in range(1, max_i, step):
            test_rec = all_records[i]
            train_records = all_records[i + 1: i + 800]
            if len(train_records) < 30:
                continue

            window = _build_window(train_records, 730)
            if len(window) < 20:
                window = train_records

            if is_lotto:
                actual_front = set(test_rec["front"])
                pool = self.cfg["front_pool"]
                front_sets = [set(r["front"]) for r in window]
                front_stats = _per_number_stats(pool, front_sets, len(window), 30, 50)

                # 各评分器输出
                ml = MLScorer(self.game, model_type="gbm")
                ml.fit(train_records)
                ml_score = ml.predict(front_stats, pool, len(window))
                trad_score = _score_map(front_stats)
                arima_score = arima_scores_lotto(window, pool)

                scores_dict = {
                    "ml_gbm": ml_score,
                    "traditional": trad_score,
                    "arima": arima_score,
                }

                # 自适应融合
                adapt_score = self._blend_adaptive(scores_dict, pool)
                adapt_ranked = sorted(adapt_score, key=lambda n: -adapt_score.get(n, 0))[:7]
                adapt_hits += len(set(adapt_ranked) & actual_front)
                adapt_picks += 7

                # 固定权重融合
                ens = StackingEnsemble(self.game)
                front_fused, _ = ens.score_lotto(train_records)
                fixed_ranked = sorted(front_fused, key=lambda n: -front_fused.get(n, 0))[:7]
                fixed_hits += len(set(fixed_ranked) & actual_front)
                fixed_picks += 7

                # 更新自适应权重
                self.update_weights(scores_dict, actual_front, top_k=7)

            else:
                positions = self.cfg["positions"]
                actual = list(test_rec["front"])
                pool = self.cfg["digit_pool"]

                for pos in range(positions):
                    seq = []
                    for r in window:
                        digits = r["front"]
                        seq.append({digits[pos]} if pos < len(digits) else set())
                    stats = _per_number_stats(pool, seq, len(window), 30, 50)

                    trad_score = _score_map(stats)
                    arima_all = arima_scores_digit(window, positions)
                    arima_score = arima_all[pos]

                    scores_dict = {
                        "traditional": trad_score,
                        "arima": arima_score,
                        "ml_gbm": {d: 0.1 for d in pool},
                    }

                    adapt_score = self._blend_adaptive(scores_dict, pool)
                    adapt_best = max(adapt_score, key=adapt_score.get)
                    adapt_hits += (1 if adapt_best == actual[pos] else 0)
                    adapt_picks += 1

                    trad_best = max(trad_score, key=trad_score.get)
                    fixed_hits += (1 if trad_best == actual[pos] else 0)
                    fixed_picks += 1

                # 更新（digit 整体更新一次）
                digit_scores = {}
                for pos in range(positions):
                    seq = []
                    for r in window:
                        digits = r["front"]
                        seq.append({digits[pos]} if pos < len(digits) else set())
                    stats = _per_number_stats(pool, seq, len(window), 30, 50)
                    digit_scores[f"pos{pos}_trad"] = _score_map(stats)
                # 简化：用总命中反馈
                total_hits = adapt_hits
                for s in self.scorer_names:
                    perf = total_hits / max(1, adapt_picks) if adapt_picks else 0.5
                    self.recent_performance[s] = self.decay * self.recent_performance[s] + (1 - self.decay) * perf

            weight_history.append(dict(self.weights))
            n_points += 1

        adapt_rate = adapt_hits / adapt_picks if adapt_picks else 0
        fixed_rate = fixed_hits / fixed_picks if fixed_picks else 0
        lift = (adapt_rate - fixed_rate) / fixed_rate if fixed_rate > 0 else 0

        return {
            "adaptive_hit_rate": round(adapt_rate, 4),
            "fixed_hit_rate": round(fixed_rate, 4),
            "lift": round(lift, 4),
            "n_points": n_points,
            "final_weights": {s: round(w, 3) for s, w in self.weights.items()},
            "weight_convergence": weight_history[-1] if weight_history else {},
        }

    def _blend_adaptive(self, scores_dict: dict, pool: list) -> Dict[int, float]:
        """用当前自适应权重融合各评分器。"""
        # 归一化
        normalized = {}
        for name, scores in scores_dict.items():
            vals = list(scores.values())
            mn, mx = min(vals), max(vals)
            if mx - mn < 1e-9:
                normalized[name] = {n: 0.5 for n in scores}
            else:
                normalized[name] = {n: (v - mn) / (mx - mn) for n, v in scores.items()}

        # 加权
        result = {}
        for n in pool:
            total = 0.0
            w_sum = 0.0
            for name in self.scorer_names:
                w = self.weights.get(name, 1.0 / len(self.scorer_names))
                s = normalized.get(name, {}).get(n, 0.0)
                total += w * s
                w_sum += w
            result[n] = total / w_sum if w_sum > 0 else 0.0
        return result
