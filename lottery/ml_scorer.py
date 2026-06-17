# -*- coding: utf-8 -*-
"""ML 评分引擎 — 用梯度提升树学习号码出现概率，替代手工加权评分。

核心思路：
  彩票号码的出现是独立随机事件，但不同号码在不同统计特征下
  的出现频率可能有微弱的非均匀性。ML 模型可以：
  1. 捕获手工评分公式忽略的非线性特征交互
  2. 自动从历史数据中学习最优特征权重
  3. 输出校准后的出现概率作为评分

使用方式：
  训练: ml = MLScorer(game)  → ml.fit(window_records)
  预测: scores = ml.predict(stats)  → {number: probability}
  回测: ml.backtest(n_test=60)  → 与传统评分对比命中率

⚠️ 彩票本质是独立随机事件——ML 不能真正"预测"。
   本模块的价值是自动特征工程 + 诚实对比。
"""

import math
from typing import Dict, List, Tuple
import numpy as np

from config import GAMES
from .analyzer import _per_number_stats, _build_window
from . import models


class MLScorer:
    """用梯度提升树学习号码出现概率。"""

    def __init__(self, game: str, model_type: str = "gbm"):
        """初始化 ML 评分器。

        game: 玩法代码（dlt/pl3/pl5）
        model_type: "gbm"(梯度提升) | "rf"(随机森林) | "lr"(逻辑回归)
        """
        self.game = game
        self.cfg = GAMES[game]
        self.model_type = model_type
        self.models = {}  # {position_or_key: sklearn_model}
        self.feature_names = [
            "freq", "omission_norm", "avg_omission_norm",
            "omission_ratio", "momentum", "ewma_freq",
            "bayes_freq", "omission_pct", "cold_hot_score",
            "freq_omission_interact",  # 频率×遗漏交互
            "momentum_ewma_interact",  # 动量×EWMA交互
        ]

    def _build_features(self, stats_dict: Dict[int, object],
                         pool: list, total_draws: int) -> Tuple[np.ndarray, list]:
        """从 NumberStat 字典构建特征矩阵。

        返回 (X, numbers) — X 每行是一个号码的特征向量。
        标签由外部提供（不在特征内部生成）。
        """
        X_rows = []
        numbers = []

        max_omission = max((s.omission for s in stats_dict.values()), default=1)
        max_avg = max((s.avg_omission for s in stats_dict.values()), default=1)

        for n in sorted(stats_dict):
            s = stats_dict[n]
            freq = s.freq
            omission_norm = s.omission / max(1, max_omission)
            avg_omission_norm = s.avg_omission / max(1, max_avg)
            omission_ratio = min(s.omission_ratio, 5.0) / 5.0
            momentum = max(-0.5, min(0.5, s.momentum)) + 0.5
            ewma = s.ewma_freq
            bayes = s.bayes_freq
            omp = s.omission_pct
            ch_map = {"热": 0.8, "平": 0.5, "冷": 0.2}
            cold_hot_score = ch_map.get(s.cold_hot, 0.5)
            freq_omission_interact = freq * (1 - omp)
            momentum_ewma_interact = momentum * ewma

            features = [
                freq, omission_norm, avg_omission_norm,
                omission_ratio, momentum, ewma,
                bayes, omp, cold_hot_score,
                freq_omission_interact, momentum_ewma_interact,
            ]

            X_rows.append(features)
            numbers.append(n)

        return np.array(X_rows), numbers

    def fit(self, train_records: list, next_draw_front: list = None,
            next_draw_back: list = None, half_life: int = 25):
        """用训练记录（最新在前）训练 ML 模型。

        train_records: 训练用开奖记录（不包含测试期）
        next_draw_front: 下一期实际开出的前区号码（用于生成标签）
        next_draw_back: 下一期实际开出的后区号码
        half_life: EWMA 半衰期
        """
        if self.cfg["type"] == "lotto":
            self._fit_lotto(train_records, next_draw_front, next_draw_back, half_life)
        else:
            self._fit_digit(train_records, next_draw_front, half_life)

    def _fit_lotto(self, train_records: list, next_front: list,
                    next_back: list, half_life: int):
        """训练大乐透模型。"""
        window = _build_window(train_records, 730)
        if len(window) < 20:
            window = train_records

        next_front_set = set(next_front) if next_front else set()
        next_back_set = set(next_back) if next_back else set()

        # 前区
        front_sets = [set(r["front"]) for r in window]
        front_stats = _per_number_stats(
            self.cfg["front_pool"], front_sets, len(window), 30, 50, half_life=half_life)
        X_f, numbers_f = self._build_features(front_stats, self.cfg["front_pool"], len(window))
        # 正确标签：号码是否在下一期出现
        y_f = np.array([1 if n in next_front_set else 0 for n in numbers_f])
        if len(set(y_f)) >= 2 and len(y_f) >= 5:
            self.models["front"] = self._train_model(X_f, y_f)
        else:
            self.models["front"] = {"type": "uniform"}

        # 后区
        back_sets = [set(r["back"]) for r in window]
        back_stats = _per_number_stats(
            self.cfg["back_pool"], back_sets, len(window), 30, 50, half_life=half_life)
        X_b, numbers_b = self._build_features(back_stats, self.cfg["back_pool"], len(window))
        y_b = np.array([1 if n in next_back_set else 0 for n in numbers_b])
        if len(set(y_b)) >= 2 and len(y_b) >= 5:
            self.models["back"] = self._train_model(X_b, y_b)
        else:
            self.models["back"] = {"type": "uniform"}

    def _fit_digit(self, train_records: list, next_digits: list, half_life: int):
        """训练排列玩法模型。"""
        window = _build_window(train_records, 730)
        if len(window) < 20:
            window = train_records

        positions = self.cfg["positions"]
        pool = self.cfg["digit_pool"]

        for pos in range(positions):
            seq = []
            for r in window:
                digits = r["front"]
                seq.append({digits[pos]} if pos < len(digits) else set())
            stats = _per_number_stats(pool, seq, len(window), 30, 50, half_life=half_life)
            X, numbers = self._build_features(stats, pool, len(window))
            # 标签：该位数字是否匹配
            target_digit = next_digits[pos] if next_digits and pos < len(next_digits) else None
            y = np.array([1 if n == target_digit else 0 for n in numbers])
            if len(set(y)) >= 2 and len(y) >= 5:
                self.models[f"pos{pos}"] = self._train_model(X, y)
            else:
                self.models[f"pos{pos}"] = {"type": "uniform"}

    def _train_model(self, X: np.ndarray, y: np.ndarray):
        """训练单个模型（v2: 增强正则化防过拟合）。"""
        if len(X) < 10 or len(set(y)) < 2:
            return {"type": "uniform"}

        try:
            if self.model_type == "gbm":
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(
                    n_estimators=30, max_depth=2, learning_rate=0.05,
                    min_samples_leaf=5, min_samples_split=10,
                    subsample=0.7, random_state=42)
            elif self.model_type == "rf":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=4,
                    min_samples_leaf=5, min_samples_split=10,
                    random_state=42)
            elif self.model_type == "lr":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(
                    C=0.1, max_iter=1000, random_state=42,
                    penalty="l2")
            else:
                return {"type": "uniform"}

            model.fit(X, y)
            return {"type": "sklearn", "model": model}
        except Exception:
            return {"type": "uniform"}

    def predict(self, stats_dict: Dict[int, object],
                pool: list, total_draws: int) -> Dict[int, float]:
        """用训练好的模型预测每个号码的出现概率。

        返回 {number: probability}，概率已校准到 [0, 1]。
        """
        X, numbers = self._build_features(stats_dict, pool, total_draws)

        model_wrapper = None
        for k in self.models:
            if self.models[k].get("type") != "uniform":
                model_wrapper = self.models[k]
                break

        if model_wrapper is None or model_wrapper["type"] == "uniform":
            # 退化为均匀分布
            nn = max(1, len(numbers))
            return {num: 1.0 / nn for num in numbers}

        model = model_wrapper["model"]
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                # probs[:, 1] 是类别1（出现）的概率
                scores = {numbers[i]: float(probs[i][1]) for i in range(len(numbers))
                          if i < len(probs)}
            else:
                preds = model.predict(X)
                scores = {numbers[i]: float(preds[i]) for i in range(len(numbers))}
        except Exception:
            n = len(numbers)
            return {n: 1.0 / n for n in numbers}

        # 校准：确保概率分布合理（不能全是0或全是1）
        vals = list(scores.values())
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-9:
            n = len(numbers)
            return {n: 1.0 / n for n in numbers}

        return scores

    def predict_lotto(self, front_stats, back_stats, total_draws: int) -> Tuple[Dict[int, float], Dict[int, float]]:
        """大乐透专用：返回 (前区分, 后区分)。"""
        front_score = self.predict(front_stats, self.cfg["front_pool"], total_draws)
        back_score = self.predict(back_stats, self.cfg["back_pool"], total_draws)
        return front_score, back_score

    def predict_digit(self, position_stats, total_draws: int) -> Dict[int, Dict[int, float]]:
        """排列玩法专用：返回 {pos: {digit: score}}。"""
        pool = self.cfg["digit_pool"]
        result = {}
        for pos, stats in position_stats.items():
            result[pos] = self.predict(stats, pool, total_draws)
        return result

    def backtest(self, n_test: int = 60, half_life: int = 25) -> dict:
        """用 ML 评分做逐期回测，与传统评分对比。

        关键改进：对每个测试点，用该点之前的所有历史窗口构建训练集，
        确保训练和预测的时间顺序正确（无未来泄漏）。

        返回 {ml_hit_rate, traditional_hit_rate, lift, n_points}
        """
        from .recommender import _score_map

        all_records = models.fetch_draws(self.game, limit=3000, order_desc=True)
        if len(all_records) < n_test + 100:
            n_test = max(5, len(all_records) - 100)
        if len(all_records) < 60:
            return {"error": "数据不足"}

        is_lotto = self.cfg["type"] == "lotto"
        ml_hits = trad_hits = 0
        ml_picks = trad_picks = 0
        n_points = 0

        step = max(1, n_test // 80)
        max_i = min(n_test, len(all_records) - 50)

        for i in range(1, max_i, step):
            test_rec = all_records[i]
            # 训练期：i+1 到 i+800（严格在测试期之前，时间上更早）
            train_start = i + 1
            train_end = min(i + 800, len(all_records))
            train_records = all_records[train_start: train_end]
            if len(train_records) < 30:
                continue

            # 从训练期内采样多个窗口构建训练集
            X_train_list = []
            y_train_list = []
            n_training_windows = min(30, len(train_records) - 40)

            for j in range(20, min(len(train_records) - 2, 20 + n_training_windows * 3), 3):
                # j 是训练窗口的起点（在 train_records 中的索引）
                # train_records[j:] 是 j 之前的数据（更早的）
                sub_train = train_records[j:]
                sub_test = train_records[j - 1]  # 紧挨着的下一期
                if len(sub_train) < 20:
                    continue

                sub_window = _build_window(sub_train, 730)
                if len(sub_window) < 20:
                    sub_window = sub_train

                if is_lotto:
                    front_sets = [set(r["front"]) for r in sub_window]
                    f_stats = _per_number_stats(
                        self.cfg["front_pool"], front_sets, len(sub_window),
                        30, 50, half_life=half_life)
                    X_f, numbers_f = self._build_features(
                        f_stats, self.cfg["front_pool"], len(sub_window))
                    sub_next = set(sub_test["front"])
                    y_f = np.array([1 if n in sub_next else 0 for n in numbers_f])
                    X_train_list.append(X_f)
                    y_train_list.append(y_f)
                else:
                    positions = self.cfg["positions"]
                    pool = self.cfg["digit_pool"]
                    for pos in range(positions):
                        seq = []
                        for r in sub_window:
                            digits = r["front"]
                            seq.append({digits[pos]} if pos < len(digits) else set())
                        stats = _per_number_stats(
                            pool, seq, len(sub_window), 30, 50, half_life=half_life)
                        X_p, numbers_p = self._build_features(stats, pool, len(sub_window))
                        target = sub_test["front"][pos] if pos < len(sub_test["front"]) else None
                        y_p = np.array([1 if n == target else 0 for n in numbers_p])
                        X_train_list.append(X_p)
                        y_train_list.append(y_p)

            if not X_train_list:
                continue

            # 合并所有训练窗口
            X_all = np.vstack(X_train_list)
            y_all = np.concatenate(y_train_list)

            if len(set(y_all)) < 2 or len(y_all) < 20:
                continue

            # 训练模型
            model = self._train_model(X_all, y_all)

            # 用当前训练窗口的特征预测测试期
            train_window = _build_window(train_records[:200], 730)
            if len(train_window) < 20:
                train_window = train_records[:200]

            if is_lotto:
                actual_front = set(test_rec["front"])
                pool = self.cfg["front_pool"]
                front_sets = [set(r["front"]) for r in train_window]
                front_stats = _per_number_stats(
                    pool, front_sets, len(train_window), 30, 50, half_life=half_life)

                ml_score = self._predict_with_model(
                    model, front_stats, pool, len(train_window))
                ml_ranked = sorted(ml_score, key=lambda n: -ml_score.get(n, 0))[:7]
                ml_hits += len(set(ml_ranked) & actual_front)
                ml_picks += 7

                trad_score = _score_map(front_stats)
                trad_ranked = sorted(trad_score, key=lambda n: -trad_score.get(n, 0))[:7]
                trad_hits += len(set(trad_ranked) & actual_front)
                trad_picks += 7
            else:
                positions = self.cfg["positions"]
                actual = list(test_rec["front"])
                pool = self.cfg["digit_pool"]

                for pos in range(positions):
                    seq = []
                    for r in train_window:
                        digits = r["front"]
                        seq.append({digits[pos]} if pos < len(digits) else set())
                    stats = _per_number_stats(
                        pool, seq, len(train_window), 30, 50, half_life=half_life)

                    ml_score = self._predict_with_model(model, stats, pool, len(train_window))
                    ml_best = max(ml_score, key=ml_score.get)
                    ml_hits += (1 if ml_best == actual[pos] else 0)
                    ml_picks += 1

                    trad_score = _score_map(stats)
                    trad_best = max(trad_score, key=trad_score.get)
                    trad_hits += (1 if trad_best == actual[pos] else 0)
                    trad_picks += 1

            n_points += 1

        ml_rate = ml_hits / ml_picks if ml_picks else 0
        trad_rate = trad_hits / trad_picks if trad_picks else 0
        lift = (ml_rate - trad_rate) / trad_rate if trad_rate > 0 else 0

        return {
            "ml_hit_rate": round(ml_rate, 4),
            "traditional_hit_rate": round(trad_rate, 4),
            "lift": round(lift, 4),
            "ml_hits": ml_hits,
            "trad_hits": trad_hits,
            "total_picks": ml_picks,
            "n_points": n_points,
        }

    def _predict_with_model(self, model_wrapper, stats_dict, pool, total_draws):
        """用给定的模型做预测。"""
        if model_wrapper is None or model_wrapper.get("type") == "uniform":
            n = len(pool)
            return {n: 1.0 / n for n in pool}

        X, numbers = self._build_features(stats_dict, pool, total_draws)
        model = model_wrapper.get("model")
        if model is None:
            n = len(numbers)
            return {n: 1.0 / n for n in numbers}

        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                scores = {numbers[i]: float(probs[i][1]) for i in range(len(numbers))
                          if i < len(probs)}
            else:
                preds = model.predict(X)
                scores = {numbers[i]: float(preds[i]) for i in range(len(numbers))}
        except Exception:
            n = len(numbers)
            return {n: 1.0 / n for n in numbers}

        vals = list(scores.values())
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-9:
            n = len(numbers)
            return {n: 1.0 / n for n in numbers}
        return scores
