# -*- coding: utf-8 -*-
"""回测引擎 — 用历史数据逐期验证推荐模型的有效性。

核心思想：
  把历史数据按时间切分。对每一个"测试点" t：
    1. 只用 t 之前的数据计算统计特征（模拟"当时"能知道的信息，避免未来泄漏）
    2. 用当前推荐模型在 t 处生成推荐
    3. 与 t 期实际开奖对比，记录命中数
  最后把模型的命中率与"纯随机基线"对比，得到诚实的有效性结论。

⚠️ 重要：若模型命中率 ≈ 随机基线，则说明模型无显著预测能力——这是正常的，
   因为彩票是独立随机事件。本引擎的价值是给出这个诚实的数字，而不是造假。
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime

from config import GAMES, STATS_WINDOW_DAYS
from . import models
from .analyzer import _build_window, _per_number_stats, _combo_constraints
from .recommender import recommend


def _binom_pvalue(hits: int, trials: int, p_null: float) -> float:
    """计算二项检验的双侧 p 值。

    H0: 每次试验命中概率 = p_null（随机基线）
    使用正态近似（trials 足够大时）。
    """
    if trials == 0 or p_null <= 0 or p_null >= 1:
        return 1.0
    n = trials
    p = p_null
    mean = n * p
    std = math.sqrt(n * p * (1 - p))
    if std < 1e-9:
        return 1.0
    # 观测命中数
    obs = hits
    # 连续性校正的 z 值
    z = abs(obs - mean - (0.5 if obs > mean else -0.5)) / std
    # 双侧 p 值（正态分布 CDF 近似）
    p_val = 2.0 * (1.0 - _normal_cdf(z))
    return max(0.0, min(1.0, p_val))


def _normal_cdf(z: float) -> float:
    """标准正态分布的 CDF（Abramowitz & Stegun 近似）。"""
    if z < -8:
        return 0.0
    if z > 8:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * math.exp(-z * z / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (
            1.781477937 + t * (-1.821255978 + t * 1.330274429)))))
    return 1.0 - p if z > 0 else p


def _binom_ci(hits: int, trials: int, alpha: float = 0.05) -> tuple:
    """Wilson score 置信区间（比正态近似更稳健）。返回 (下限, 上限)。"""
    if trials == 0:
        return (0.0, 1.0)
    p_hat = hits / trials
    z = 1.96  # alpha=0.05 对应的 z 值
    n = trials
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def _chi_square_test(obs_hits: int, obs_miss: int, exp_rate: float, total: int) -> dict:
    """卡方检验：观测命中数 vs 期望命中数。

    返回 {'chi2': float, 'p_value': float, 'significant_005': bool}
    """
    exp_hits = total * exp_rate
    exp_miss = total * (1 - exp_rate)
    if exp_hits < 1 or exp_miss < 1:
        return {"chi2": 0.0, "p_value": 1.0, "significant_005": False}
    chi2 = ((obs_hits - exp_hits) ** 2 / exp_hits +
            (obs_miss - exp_miss) ** 2 / exp_miss)
    # 1 自由度的卡方 p 值近似
    if chi2 < 0.01:
        p_val = 1.0
    else:
        # 用正态近似：sqrt(2*chi2) - sqrt(2*df-1) ≈ N(0,1)
        z = math.sqrt(2 * chi2) - math.sqrt(2 * 1 - 1)
        p_val = 2 * (1 - _normal_cdf(abs(z)))  # 双侧
    p_val = max(0.0, min(1.0, p_val))
    return {"chi2": round(chi2, 4), "p_value": round(p_val, 4),
            "significant_005": p_val < 0.05}


@dataclass
class BacktestPoint:
    """单个测试点的命中情况。"""
    issue: str
    draw_date: str
    # 模型推荐 top-k 个号（前区/主号）
    predicted_topk: List[int] = field(default_factory=list)
    actual: List[int] = field(default_factory=list)
    hits: int = 0            # 实际命中数
    pick_size: int = 0       # 推荐了多少个（用于算命中率）
    pool_size: int = 0       # 号池大小（用于算随机基线）


@dataclass
class BacktestResult:
    game: str
    name: str
    points: List[BacktestPoint] = field(default_factory=list)
    model_hit_rate: float = 0.0     # 模型平均命中率 = Σhits / (Σpick_size)
    random_hit_rate: float = 0.0    # 随机基线命中率 = pick_size/pool_size 的平均
    lift: float = 0.0               # (model - random) / random，>0 表示优于随机
    verdict: str = ""               # 人类可读结论
    # 统计显著性检验
    p_value: float = 1.0            # 二项检验双侧 p 值
    ci_lower: float = 0.0           # 模型命中率 95% 置信区间下限
    ci_upper: float = 1.0           # 模型命中率 95% 置信区间上限
    chi2: float = 0.0               # 卡方统计量
    chi2_p: float = 1.0             # 卡方检验 p 值
    is_significant: bool = False    # p < 0.05?

    def summarize(self, use_random: bool = False):
        if not self.points:
            return
        total_hits = sum(p.hits for p in self.points)
        total_picks = sum(p.pick_size for p in self.points) or 1
        self.model_hit_rate = total_hits / total_picks
        if not use_random:
            # 用每点 pick_size/pool_size 估算随机基线
            rand_rates = [p.pick_size / p.pool_size for p in self.points if p.pool_size]
            self.random_hit_rate = sum(rand_rates) / len(rand_rates) if rand_rates else 0
        # 否则 random_hit_rate 已由调用方设置为理论值
        if self.random_hit_rate > 1e-9:
            self.lift = (self.model_hit_rate - self.random_hit_rate) / self.random_hit_rate

        # ---- 统计显著性检验 ----
        # 二项检验：每次选号视为一次伯努利试验，命中概率 = random_rate
        total_trials = total_picks  # 总选号次数
        self.p_value = _binom_pvalue(total_hits, total_trials, self.random_hit_rate)
        # Wilson 置信区间
        self.ci_lower, self.ci_upper = _binom_ci(total_hits, total_trials, alpha=0.05)
        # 卡方检验
        total_miss = total_trials - total_hits
        chi2_res = _chi_square_test(total_hits, total_miss, self.random_hit_rate, total_trials)
        self.chi2 = chi2_res["chi2"]
        self.chi2_p = chi2_res["p_value"]
        self.is_significant = chi2_res["significant_005"]

        # 结论（阈值放宽，彩票噪声大，±10% 内都算"无差别"）
        sig_text = ""
        if self.is_significant:
            if self.lift > 0:
                sig_text = f"统计显著优于随机(p={self.p_value:.3f})，但彩票噪声极大，不可外推。"
            else:
                sig_text = f"统计显著劣于随机(p={self.p_value:.3f})，策略有害。"
        else:
            sig_text = f"统计不显著(p={self.p_value:.3f})。"

        if self.lift > 0.10:
            self.verdict = f"模型命中率略高于随机基线（+{self.lift:.1%}）。{sig_text}"
        elif self.lift > -0.10:
            self.verdict = f"模型命中率与纯随机基线无显著差别（{self.lift:+.1%}）。{sig_text}这符合彩票独立随机的本质——无可靠预测可言。"
        else:
            self.verdict = f"模型命中率低于随机基线（{self.lift:.1%}）。{sig_text}当前策略有害，应调整为更分散/随机的方式。"


def _model_pick_topk(game: str, window_records, k: int) -> List[int]:
    """用 window_records（t 之前的数据）模拟当时的模型，返回 top-k 候选号。

    lotto: 返回前区综合分最高的 k 个号。
    digit: 不使用此函数（digit 按位回测，见 _digit_position_pick）。
    """
    from .recommender import _score_map
    cfg = GAMES[game]
    seq = [set(r["front"]) for r in window_records]
    stats = _per_number_stats(cfg["front_pool"], seq, len(seq), 30, 50)
    score = _score_map(stats)
    ranked = sorted(score, key=lambda n: -score.get(n, 0))
    return ranked[:k]


def _digit_position_pick(window_records, positions: int) -> List[int]:
    """digit 玩法按位预测：返回每位的 top-1 数字。"""
    from .recommender import _score_map
    cfg_pool = list(range(0, 10))
    picks = []
    for pos in range(positions):
        seq = []
        for r in window_records:
            digits = r["front"]
            seq.append({digits[pos]} if pos < len(digits) else set())
        stats = _per_number_stats(cfg_pool, seq, len(seq), 30, 50)
        score = _score_map(stats)
        ranked = sorted(score, key=lambda n: -score.get(n, 0))
        picks.append(ranked[0] if ranked else 0)
    return picks


def _actual_front(game: str, record: dict) -> List[int]:
    """返回某期实际开出的前区/主号（lotto 用）。"""
    return list(record["front"])


def backtest(game: str, n_test: int = 100, window_days: int = STATS_WINDOW_DAYS) -> BacktestResult:
    """对某玩法做逐期回测。

    n_test: 测试最近多少期
    window_days: 每个测试点用之前多少天的数据做特征
    """
    cfg = GAMES[game]
    res = BacktestResult(game=game, name=cfg["name"])
    # 取足够多的历史（测试点 + 训练窗口），最新在前
    all_records = models.fetch_draws(game, limit=3000, order_desc=True)
    if len(all_records) < n_test + 50:
        n_test = max(5, len(all_records) - 50)
    if len(all_records) < 20:
        return res

    # 测试点：从较新的位置往前，倒序回放
    is_lotto = cfg["type"] == "lotto"
    if is_lotto:
        pick_size = 7                          # 推荐前区 7 个号
        pool_size = len(cfg["front_pool"])     # 35
        # 随机基线：从 35 选 7，每期开 5 个，期望命中 = 7*5/35 = 1 个 → 命中率 1/7
        random_rate = pick_size * 5 / pool_size / pick_size   # = 5/35
    else:
        positions = cfg["positions"]
        pick_size = positions                  # 每位预测 1 个，共 positions 个
        pool_size = 10                         # 每位号池 0-9
        # 随机基线：每位 1/10 命中
        random_rate = 1.0 / pool_size

    step = max(1, n_test // 80)  # 采样密度：尽量多取测试点提升统计稳定性
    max_i = min(n_test, len(all_records) - 30)
    for i in range(1, max_i, step):
        test_rec = all_records[i]
        # 训练窗口：i 之后的记录（更早的，时间上在 t 之前）
        train_records = all_records[i + 1: i + 800]
        if len(train_records) < 30:
            continue
        train_window = _build_window(train_records, window_days)
        if len(train_window) < 20:
            train_window = train_records

        if is_lotto:
            predicted = _model_pick_topk(game, train_window, pick_size)
            actual = _actual_front(game, test_rec)
            hits = len(set(predicted) & set(actual))
        else:
            predicted = _digit_position_pick(train_window, positions)
            actual = list(test_rec["front"])
            # 按位比较：位置相同才算命中
            hits = sum(1 for j in range(min(len(predicted), len(actual)))
                       if predicted[j] == actual[j])

        res.points.append(BacktestPoint(
            issue=test_rec["issue"],
            draw_date=test_rec["draw_date"][:10] if test_rec["draw_date"] else "",
            predicted_topk=predicted, actual=actual,
            hits=hits, pick_size=pick_size, pool_size=pool_size,
        ))
    # 用理论随机基线（更准确），覆盖 summarize 里的估算
    res.random_hit_rate = random_rate
    res.summarize(use_random=True)
    return res


def backtest_all(n_test: int = 100) -> Dict[str, BacktestResult]:
    return {g: backtest(g, n_test) for g in ("dlt", "pl3", "pl5")}


def backtest_as_dict(res: BacktestResult) -> dict:
    """转成前端友好的 dict。"""
    return {
        "game": res.game, "name": res.name,
        "n_points": len(res.points),
        "model_hit_rate": round(res.model_hit_rate, 4),
        "random_hit_rate": round(res.random_hit_rate, 4),
        "lift": round(res.lift, 4),
        "verdict": res.verdict,
        "p_value": round(res.p_value, 4),
        "ci_lower": round(res.ci_lower, 4),
        "ci_upper": round(res.ci_upper, 4),
        "chi2": res.chi2,
        "chi2_p": round(res.chi2_p, 4),
        "is_significant": res.is_significant,
        "recent": [
            {"issue": p.issue, "date": p.draw_date,
             "predicted": p.predicted_topk, "actual": p.actual,
             "hits": p.hits, "pick_size": p.pick_size, "pool_size": p.pool_size}
            for p in res.points[:15]
        ],
    }


if __name__ == "__main__":
    models.init_db()
    print("逐期回测中（这可能需要几十秒）...\n")
    results = backtest_all(n_test=120)
    for g, res in results.items():
        print(f"===== {res.name} =====")
        print(f"  测试点数: {len(res.points)}")
        print(f"  模型命中率: {res.model_hit_rate:.2%}  (推荐号中实际开出的比例)")
        print(f"  随机基线:   {res.random_hit_rate:.2%}  (纯随机选号的期望命中率)")
        print(f"  提升:       {res.lift:+.2%}")
        print(f"  结论: {res.verdict}\n")
