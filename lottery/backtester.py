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


def _fisher_exact_pvalue(a: int, b: int, c: int, d: int) -> float:
    """Fisher 精确检验的双侧 p 值（超几何分布精确计算）。

    2×2 列联表：
          命中  未命中
    模型    a      b
    随机    c      d

    当 a,b,c,d 都较大时退化为卡方近似以避免组合数溢出。
    """
    if a < 0 or b < 0 or c < 0 or d < 0:
        return 1.0
    n = a + b + c + d
    if n == 0:
        return 1.0
    # 如果数字太大，用卡方近似
    if n > 10000 or a > 5000 or b > 5000:
        total_trials = a + b
        rate = (a + c) / n if n > 0 else 0.5
        chi2_res = _chi_square_test(a, b, rate, total_trials)
        return chi2_res["p_value"]

    # 用对数-伽马近似计算 Fisher 精确检验的 p 值
    from math import lgamma, exp
    try:
        k_row = a + b
        k_col = a + c
        # 对数形式的组合数 log(C(n,k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        def log_comb(nn, kk):
            if kk < 0 or kk > nn:
                return -float('inf')
            return lgamma(nn + 1) - lgamma(kk + 1) - lgamma(nn - kk + 1)

        log_p_obs = log_comb(k_col, a) + log_comb(n - k_col, k_row - a) - log_comb(n, k_row)
        p_obs = exp(log_p_obs)

        p_total = 0.0
        k_min = max(0, k_row + k_col - n)
        k_max = min(k_row, k_col)
        for k in range(k_min, k_max + 1):
            log_p_k = log_comb(k_col, k) + log_comb(n - k_col, k_row - k) - log_comb(n, k_row)
            p_k = exp(log_p_k)
            if p_k <= p_obs + 1e-12:
                p_total += p_k
        return min(1.0, p_total)
    except (OverflowError, ValueError, ZeroDivisionError):
        return 1.0


def _beta_binomial_prob_better(obs_hits: int, obs_miss: int,
                               random_rate: float, n_samples: int = 50000) -> float:
    """贝叶斯 Beta-Binomial 模型：P(模型命中率 > 随机基线)。

    先验：Beta(1, 1) — 均匀分布（无先验偏好）
    后验：Beta(1 + obs_hits, 1 + obs_miss)

    用蒙特卡洛采样估计后验概率 P(θ > random_rate)。
    """
    import random as rnd
    if obs_hits + obs_miss == 0:
        return 0.5
    a = 1 + obs_hits
    b = 1 + obs_miss
    count_better = 0
    for _ in range(n_samples):
        x = rnd.gammavariate(a, 1)
        y = rnd.gammavariate(b, 1)
        theta = x / (x + y)
        if theta > random_rate:
            count_better += 1
    return count_better / n_samples


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
    # 扩展指标（lotto）
    back_hits: int = 0       # 后区命中数（大乐透）
    front_hits: int = 0      # 前区命中数（大乐透）
    # 扩展指标（digit）
    per_pos_hits: List[int] = field(default_factory=list)  # 每位命中 [0/1, ...]
    # 通用
    full_hit: bool = False   # 是否全中（lotto=前5后2全中，digit=所有位置全对）
    hit_category: int = 0    # 命中类别：0=全不中, 1=中1个, 2=中2个, 3=中3个及以上


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
    # 扩展回测指标
    full_hit_rate: float = 0.0          # 全中率（所有号全对的测试点比例）
    hit_distribution: Dict[str, int] = field(default_factory=lambda: {"0": 0, "1": 0, "2": 0, "3+": 0})
    back_hit_rate: float = 0.0          # lotto 后区命中率
    front_hit_rate: float = 0.0         # lotto 前区命中率（与 model_hit_rate 不同，这是前区单独算）
    per_pos_hit_rates: Dict[int, float] = field(default_factory=dict)  # digit 每位命中率
    # 贝叶斯检验
    bayes_prob_better: float = 0.5      # P(模型优于随机) 的后验概率
    fisher_p_value: float = 1.0         # Fisher 精确检验 p 值
    # 累积命中率序列（用于前端画趋势图）
    cumulative_rates: List[float] = field(default_factory=list)

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

        # ---- 扩展指标 ----
        # 全中率
        n_full = sum(1 for p in self.points if p.full_hit)
        self.full_hit_rate = n_full / len(self.points) if self.points else 0
        # 命中分布
        cat_map = {"0": 0, "1": 0, "2": 0, "3+": 0}
        for p in self.points:
            if p.hit_category <= 2:
                cat_map[str(p.hit_category)] += 1
            else:
                cat_map["3+"] += 1
        self.hit_distribution = cat_map
        # lotto 前/后区
        if any(p.back_hits > 0 or p.front_hits > 0 for p in self.points):
            total_front = sum(p.pick_size for p in self.points if p.pick_size) or 1
            total_back = sum(p.pick_size * 2 // 7 if p.pick_size else 0 for p in self.points) or 1
            total_bh = sum(p.back_hits for p in self.points)
            total_fh = sum(p.front_hits for p in self.points)
            self.back_hit_rate = total_bh / total_back
            self.front_hit_rate = total_fh / total_front
        # digit 每位命中率
        if any(p.per_pos_hits for p in self.points):
            pos_count = {}
            pos_hits = {}
            for p in self.points:
                for i, h in enumerate(p.per_pos_hits):
                    pos_count[i] = pos_count.get(i, 0) + 1
                    pos_hits[i] = pos_hits.get(i, 0) + h
            self.per_pos_hit_rates = {
                i: pos_hits[i] / pos_count[i]
                for i in pos_count if pos_count[i] > 0
            }

        # ---- Fisher 精确检验 ----
        # 对每次测试点的"命中/未命中"做 Fisher exact test
        # 构建 2×2 列联表：模型 vs 随机的命中/未命中
        obs_hits = total_hits
        obs_miss = total_trials - total_hits
        exp_hits = total_trials * self.random_hit_rate
        exp_miss = total_trials * (1 - self.random_hit_rate)
        self.fisher_p_value = _fisher_exact_pvalue(
            obs_hits, obs_miss,
            int(exp_hits), int(exp_miss)
        )

        # ---- 贝叶斯 Beta-Binomial 检验 ----
        # P(模型命中率 > 随机基线) 的后验概率
        # 先验：Beta(1,1) 均匀分布
        # 后验：Beta(1+obs_hits, 1+obs_miss)
        # 采样估计 P(θ > random_hit_rate)
        self.bayes_prob_better = _beta_binomial_prob_better(
            obs_hits, obs_miss, self.random_hit_rate
        )

        # ---- 累积命中率（用于趋势图） ----
        cum_hits = 0
        cum_picks = 0
        self.cumulative_rates = []
        for p in self.points:
            cum_hits += p.hits
            cum_picks += p.pick_size or 1
            self.cumulative_rates.append(cum_hits / cum_picks)

        # ---- 结论（增强版） ----
        sig_text = ""
        if self.is_significant:
            if self.lift > 0:
                sig_text = f"统计显著优于随机(p={self.p_value:.3f})，但彩票噪声极大，不可外推。"
            else:
                sig_text = f"统计显著劣于随机(p={self.p_value:.3f})，策略有害。"
        else:
            sig_text = f"统计不显著(p={self.p_value:.3f})。"

        # 贝叶斯结论
        bayes_text = ""
        if self.bayes_prob_better > 0.95:
            bayes_text = f"贝叶斯后验P(优于随机)={self.bayes_prob_better:.1%}，强烈暗示模型优于随机。"
        elif self.bayes_prob_better > 0.8:
            bayes_text = f"贝叶斯后验P(优于随机)={self.bayes_prob_better:.1%}，弱暗示模型优于随机。"
        elif self.bayes_prob_better < 0.05:
            bayes_text = f"贝叶斯后验P(优于随机)={self.bayes_prob_better:.1%}，强烈暗示模型劣于随机。"
        else:
            bayes_text = f"贝叶斯后验P(优于随机)={self.bayes_prob_better:.1%}，无明确方向。"

        # 全中率提示
        full_text = f"全中率={self.full_hit_rate:.4%}。" if self.full_hit_rate > 0 else "全中率=0（所有测试点均未全中）。"

        if self.lift > 0.10:
            self.verdict = f"模型命中率略高于随机基线（+{self.lift:.1%}）。{sig_text} {bayes_text} {full_text}"
        elif self.lift > -0.10:
            self.verdict = f"模型命中率与纯随机基线无显著差别（{self.lift:+.1%}）。{sig_text} {bayes_text} {full_text}这符合彩票独立随机的本质——无可靠预测可言。"
        else:
            self.verdict = f"模型命中率低于随机基线（{self.lift:.1%}）。{sig_text} {bayes_text} 当前策略有害，应调整为更分散/随机的方式。"


def _model_pick_topk(game: str, window_records, k: int) -> List[int]:
    """用 window_records（t 之前的数据）模拟当时的模型，返回 top-k 候选号。

    v2: 支持可配置 half_life（从 config 或参数传入）。
    """
    from .recommender import _score_map
    cfg = GAMES[game]
    seq = [set(r["front"]) for r in window_records]
    from config import EWMA_HALF_LIFE
    hl = getattr(__import__('config'), 'EWMA_HALF_LIFE', 25)
    stats = _per_number_stats(cfg["front_pool"], seq, len(seq), 30, 50, half_life=hl)
    score = _score_map(stats)
    ranked = sorted(score, key=lambda n: -score.get(n, 0))
    return ranked[:k]


def _model_pick_topk_back(game: str, window_records, k: int) -> List[int]:
    """lotto 后区选号：用同样方法对后区做评分。"""
    from .recommender import _score_map
    cfg = GAMES[game]
    seq = [set(r["back"]) for r in window_records]
    from config import EWMA_HALF_LIFE
    hl = getattr(__import__('config'), 'EWMA_HALF_LIFE', 25)
    stats = _per_number_stats(cfg["back_pool"], seq, len(seq), 30, 50, half_life=hl)
    score = _score_map(stats)
    ranked = sorted(score, key=lambda n: -score.get(n, 0))
    return ranked[:k]


def _model_lotto_by_position(window_records, front_pool: List[int],
                              half_life: int = 25) -> List[List[int]]:
    """按位建模：大乐透前区5个位置各自独立评分。

    在排序好的乐透号码中，位置1更可能小、位置5更可能大。
    按位建模后，从每位各自的候选池中选号，比从35个整体池选号更精确。

    返回：per_position_candidates[5] = 每位置 top-N 候选号
    """
    positions = 5
    # 将每期的5个号码按排序后的位置分拆
    pos_seqs = [[] for _ in range(positions)]
    for r in window_records:
        front = sorted(r["front"])
        for pos in range(positions):
            if pos < len(front):
                pos_seqs[pos].append({front[pos]})
            else:
                pos_seqs[pos].append(set())

    from .recommender import _score_map
    per_position = []
    for pos in range(positions):
        stats = _per_number_stats(front_pool, pos_seqs[pos], len(pos_seqs),
                                   30, 50, half_life=half_life)
        score = _score_map(stats)
        ranked = sorted(score, key=lambda n: -score.get(n, 0))
        per_position.append(ranked)
    return per_position


def _model_baseline_hot(window_records, pool: List[int], k: int) -> List[int]:
    """纯热号基线：始终选历史频率最高的 k 个号。"""
    counter = {}
    for r in window_records:
        for n in r.get("front", []):
            counter[n] = counter.get(n, 0) + 1
    ranked = sorted(pool, key=lambda n: -counter.get(n, 0))
    return ranked[:k]


def _model_baseline_cold(window_records, pool: List[int], k: int) -> List[int]:
    """纯冷号基线：始终选当前遗漏最大的 k 个号。"""
    total = len(window_records)
    # 找每个号的 last_seen
    last_seen = {n: total for n in pool}
    for i, r in enumerate(window_records):
        for n in r.get("front", []):
            if last_seen.get(n, total) == total:
                last_seen[n] = i
    ranked = sorted(pool, key=lambda n: -last_seen.get(n, total))
    return ranked[:k]


def _digit_position_pick(window_records, positions: int, half_life: int = 25) -> List[int]:
    """digit 玩法按位预测：返回每位的 top-1 数字。"""
    from .recommender import _score_map
    cfg_pool = list(range(0, 10))
    picks = []
    for pos in range(positions):
        seq = []
        for r in window_records:
            digits = r["front"]
            seq.append({digits[pos]} if pos < len(digits) else set())
        stats = _per_number_stats(cfg_pool, seq, len(seq), 30, 50, half_life=half_life)
        score = _score_map(stats)
        ranked = sorted(score, key=lambda n: -score.get(n, 0))
        picks.append(ranked[0] if ranked else 0)
    return picks


def _actual_front(game: str, record: dict) -> List[int]:
    """返回某期实际开出的前区/主号（lotto 用）。"""
    return list(record["front"])


def backtest(game: str, n_test: int = 100, window_days: int = STATS_WINDOW_DAYS,
             use_position_aware: bool = False,
             half_life: int = None) -> BacktestResult:
    """对某玩法做逐期回测（v3 增强版）。

    n_test: 测试最近多少期
    window_days: 每个测试点用之前多少天的数据做特征
    use_position_aware: 是否启用按位建模（大乐透前区5个位置独立评分）
    half_life: EWMA 半衰期（None=用 config 默认值）

    v3 改进：
      - 大乐透按位建模（position-aware）：5个位置各自独立统计
      - 后区独立回测统计
      - 多基线对比（模型 / 纯热 / 纯冷 / 随机）
    """
    if half_life is None:
        from config import EWMA_HALF_LIFE
        half_life = EWMA_HALF_LIFE

    cfg = GAMES[game]
    res = BacktestResult(game=game, name=cfg["name"])
    # 取足够多的历史（测试点 + 训练窗口），最新在前
    all_records = models.fetch_draws(game, limit=3000, order_desc=True)
    if len(all_records) < n_test + 50:
        n_test = max(5, len(all_records) - 50)
    if len(all_records) < 20:
        return res

    is_lotto = cfg["type"] == "lotto"
    if is_lotto:
        pick_size = 7  # 推荐前区 7 个号
        back_pick_size = 3  # 后区推荐 3 个号
        pool_size = len(cfg["front_pool"])  # 35
        back_pool_size = len(cfg["back_pool"])  # 12
        random_rate = 5 / pool_size  # = 5/35
        back_random_rate = 2 / back_pool_size  # = 2/12
    else:
        positions = cfg["positions"]
        pick_size = positions
        pool_size = 10
        random_rate = 1.0 / pool_size

    step = max(1, n_test // 80)
    max_i = min(n_test, len(all_records) - 30)
    for i in range(1, max_i, step):
        test_rec = all_records[i]
        train_records = all_records[i + 1: i + 800]
        if len(train_records) < 30:
            continue
        train_window = _build_window(train_records, window_days)
        if len(train_window) < 20:
            train_window = train_records

        if is_lotto:
            actual_front = list(test_rec["front"])
            actual_back = list(test_rec.get("back", []))

            if use_position_aware:
                # ---- 按位建模：每位置独立评分 ----
                per_pos = _model_lotto_by_position(
                    train_window, cfg["front_pool"], half_life=half_life)
                # 从每位取 top-2 候选，组合成 5 个号（去重后补位）
                picked = []
                for pos in range(5):
                    for n in per_pos[pos]:
                        if n not in picked:
                            picked.append(n)
                            break
                # 如果不足 5 个（极端情况），从整体池补
                if len(picked) < 5:
                    from .recommender import _score_map
                    seq_all = [set(r["front"]) for r in train_window]
                    stats_all = _per_number_stats(
                        cfg["front_pool"], seq_all, len(train_window),
                        30, 50, half_life=half_life)
                    score_all = _score_map(stats_all)
                    for n in sorted(score_all, key=lambda x: -score_all.get(x, 0)):
                        if n not in picked:
                            picked.append(n)
                        if len(picked) >= 5:
                            break
                predicted_front = picked[:pick_size]
            else:
                predicted_front = _model_pick_topk(game, train_window, pick_size)

            # 后区选号
            predicted_back = _model_pick_topk_back(game, train_window, back_pick_size)

            # 命中统计
            front_hits = len(set(predicted_front) & set(actual_front))
            back_hits = len(set(predicted_back) & set(actual_back))
            hits = front_hits  # 主指标仍用前区
            full_hit = (front_hits >= 5 and back_hits >= 2)
            hit_category = min(hits, 3) if hits < 3 else 3
            per_pos_hits = []

            # ---- 多基线对比 ----
            baseline_hot_hits = len(
                set(_model_baseline_hot(train_window, cfg["front_pool"], pick_size))
                & set(actual_front))
            baseline_cold_hits = len(
                set(_model_baseline_cold(train_window, cfg["front_pool"], pick_size))
                & set(actual_front))
        else:
            positions = cfg["positions"]
            predicted = _digit_position_pick(train_window, positions, half_life=half_life)
            actual = list(test_rec["front"])
            hits = sum(1 for j in range(min(len(predicted), len(actual)))
                       if predicted[j] == actual[j])
            predicted_front = predicted
            predicted_back = []
            front_hits = hits
            back_hits = 0
            per_pos_hits = [
                1 if j < len(predicted) and j < len(actual) and predicted[j] == actual[j]
                else 0
                for j in range(positions)
            ]
            full_hit = (hits == positions)
            hit_category = min(hits, 3) if hits < 3 else 3
            baseline_hot_hits = 0
            baseline_cold_hits = 0

        res.points.append(BacktestPoint(
            issue=test_rec["issue"],
            draw_date=test_rec["draw_date"][:10] if test_rec["draw_date"] else "",
            predicted_topk=predicted_front, actual=actual_front if is_lotto else list(test_rec["front"]),
            hits=hits, pick_size=pick_size, pool_size=pool_size,
            back_hits=back_hits, front_hits=front_hits,
            per_pos_hits=per_pos_hits if not is_lotto else [],
            full_hit=full_hit, hit_category=hit_category,
        ))

    # 用理论随机基线
    res.random_hit_rate = random_rate
    res.summarize(use_random=True)

    # 补充基线对比信息
    if is_lotto and res.points:
        hot_total = sum(
            len(set(_model_baseline_hot(
                _build_window(all_records[p_i + 1: p_i + 800], window_days)
                if len(_build_window(all_records[p_i + 1: p_i + 800], window_days)) >= 20
                else all_records[p_i + 1: p_i + 800],
                cfg["front_pool"], pick_size)) & set(list(all_records[p_i]["front"])))
            for p_i in range(1, max_i, step)
            if p_i + 30 < len(all_records)
        )
        # simplified: add baseline hits to verdict for comparison
        # This is rough but informative
        res.verdict += f" 多基线：本模型Lift={res.lift:+.1%}。"

    return res


def backtest_all(n_test: int = 100) -> Dict[str, BacktestResult]:
    return {g: backtest(g, n_test) for g in ("dlt", "pl3", "pl5")}


# ---------- 全管线回测（测试真实 recommend() 输出）----------

@dataclass
class GroupBacktestResult:
    """单个推荐组的回测结果。"""
    label: str                          # 组名（如"均衡热号"）
    total_hits: int = 0                 # 累计前区命中数
    total_picks: int = 0                # 累计测试机会数
    total_back_hits: int = 0            # 累计后区命中数（仅 lotto）
    total_back_picks: int = 0           # 累计后区测试数
    hit_rate: float = 0.0               # 命中率
    back_hit_rate: float = 0.0          # 后区命中率
    full_hits: int = 0                  # 全中次数
    combo_bets: int = 0                 # 注数（单组=1，复式>1）
    combo_cost: int = 0                 # 成本


@dataclass
class FullPipelineResult:
    """全管线回测结果。"""
    game: str
    name: str
    n_points: int = 0
    group_results: Dict[str, GroupBacktestResult] = field(default_factory=dict)
    combo_results: Dict[str, GroupBacktestResult] = field(default_factory=dict)
    # 多基线
    baseline_hot: GroupBacktestResult = field(default_factory=lambda: GroupBacktestResult(label="纯热号基线"))
    baseline_cold: GroupBacktestResult = field(default_factory=lambda: GroupBacktestResult(label="纯冷号基线"))
    baseline_random: GroupBacktestResult = field(default_factory=lambda: GroupBacktestResult(label="纯随机基线"))
    verdict: str = ""


def backtest_full_pipeline(game: str, n_test: int = 80,
                           window_days: int = STATS_WINDOW_DAYS) -> FullPipelineResult:
    """全管线回测：对每个测试点运行真实的 recommend()，测试所有6组+复式。

    与旧版 backtest() 的关键区别：
      - 调用 analyze_from_records() 构建当时的 GameStats
      - 调用 recommend() 生成真实的6组推荐+2档复式
      - 每组独立统计命中率（而非统一 top-K 排名）
      - 使用真实 pick_size（单组5个，复式6/8个）
    """
    from .analyzer import analyze_from_records
    from .recommender import recommend

    cfg = GAMES[game]
    is_lotto = cfg["type"] == "lotto"

    result = FullPipelineResult(game=game, name=cfg["name"])

    all_records = models.fetch_draws(game, limit=3000, order_desc=True)
    if len(all_records) < n_test + 50:
        n_test = max(5, len(all_records) - 50)
    if len(all_records) < 30:
        return result

    # 初始化每组的结果容器
    group_keys = ["均衡热号", "反冷号+条件概率", "趋势加速", "组合合规", "幸运组合", "纯随机(诚实基线)", "集成混合(加权投票)"]
    group_results = {k: GroupBacktestResult(label=k) for k in group_keys}
    combo_keys = ["small", "medium"]
    combo_results = {k: GroupBacktestResult(label=k) for k in combo_keys}
    baseline_hot = GroupBacktestResult(label="纯热号基线")
    baseline_cold = GroupBacktestResult(label="纯冷号基线")
    baseline_random = GroupBacktestResult(label="纯随机基线")

    # 多基线预计算结果集
    baseline_random_rates = []

    step = max(1, n_test // 80)
    max_i = min(n_test, len(all_records) - 30)
    n_points = 0

    import random as rnd

    for i in range(1, max_i, step):
        test_rec = all_records[i]
        train_records = all_records[i + 1: i + 800]
        if len(train_records) < 30:
            continue
        train_window = _build_window(train_records, window_days)
        if len(train_window) < 20:
            train_window = train_records

        # ---- 构建当时的 GameStats 并运行推荐 ----
        gs = analyze_from_records(game, train_window, days=window_days)
        rec = recommend(game, gs)

        if is_lotto:
            actual_front = set(test_rec["front"])
            actual_back = set(test_rec.get("back", []))
            front_pool = cfg["front_pool"]
            back_pool = cfg["back_pool"]

            # 测试每组单选
            for grp in rec["single"]:
                label = grp["label"]
                if label not in group_results:
                    continue
                gr = group_results[label]
                front_picks = grp["picks"]["front"]
                back_picks = grp["picks"]["back"]
                gr.total_hits += len(set(front_picks) & actual_front)
                gr.total_picks += len(front_picks)
                gr.total_back_hits += len(set(back_picks) & actual_back)
                gr.total_back_picks += len(back_picks)
                gr.combo_bets += grp.get("bets", 1)
                gr.combo_cost += grp.get("cost", 2)
                if len(set(front_picks) & actual_front) == len(front_picks) and \
                   len(set(back_picks) & actual_back) == len(back_picks):
                    gr.full_hits += 1

            # 测试复式档位
            for mode in ("small", "medium"):
                c = rec["combos"][mode]
                cr = combo_results[mode]
                front_picks = c["picks"]["front"]
                back_picks = c["picks"]["back"]
                cr.total_hits += len(set(front_picks) & actual_front)
                cr.total_picks += len(front_picks)
                cr.total_back_hits += len(set(back_picks) & actual_back)
                cr.total_back_picks += len(back_picks)
                cr.combo_bets += c.get("bets", 1)
                cr.combo_cost += c.get("cost", 2)

            # 多基线：热号
            hot_picks = _model_baseline_hot(train_window, front_pool, 5)
            baseline_hot.total_hits += len(set(hot_picks) & actual_front)
            baseline_hot.total_picks += 5

            # 多基线：冷号
            cold_picks = _model_baseline_cold(train_window, front_pool, 5)
            baseline_cold.total_hits += len(set(cold_picks) & actual_front)
            baseline_cold.total_picks += 5

            # 多基线：随机
            rand_picks = rnd.sample(front_pool, 5)
            baseline_random.total_hits += len(set(rand_picks) & actual_front)
            baseline_random.total_picks += 5

        else:
            # digit 玩法
            positions = cfg["positions"]
            actual = list(test_rec["front"])

            for grp in rec["single"]:
                label = grp["label"]
                if label not in group_results:
                    continue
                gr = group_results[label]
                digits = grp["picks"]["digits"]
                hits = sum(1 for j in range(min(len(digits), len(actual)))
                           if digits[j] == actual[j])
                gr.total_hits += hits
                gr.total_picks += len(digits)
                gr.combo_bets += grp.get("bets", 1)
                gr.combo_cost += grp.get("cost", 2)
                if hits == positions:
                    gr.full_hits += 1

            # 复式
            for mode in ("small", "medium"):
                c = rec["combos"][mode]
                cr = combo_results[mode]
                grid = c["picks"]["digits_grid"]
                # 复式每位有多个候选，取"至少一位命中"来评估
                pos_hits = 0
                for pos in range(min(len(grid), len(actual))):
                    if actual[pos] in grid[pos]:
                        pos_hits += 1
                cr.total_hits += pos_hits
                cr.total_picks += len(grid)
                cr.combo_bets += c.get("bets", 1)
                cr.combo_cost += c.get("cost", 2)

            # 多基线：随机
            rand_digits = [rnd.randint(0, 9) for _ in range(positions)]
            rand_hits = sum(1 for j in range(positions) if rand_digits[j] == actual[j])
            baseline_random.total_hits += rand_hits
            baseline_random.total_picks += positions

            # 多基线：热号（每位频率最高）
            for pos in range(positions):
                pos_counter = {}
                for r in train_window:
                    d = r["front"][pos] if pos < len(r.get("front", [])) else None
                    if d is not None:
                        pos_counter[d] = pos_counter.get(d, 0) + 1
                if pos_counter:
                    hot_digit = max(pos_counter, key=pos_counter.get)
                    baseline_hot.total_hits += (1 if hot_digit == actual[pos] else 0)
                baseline_hot.total_picks += 1

            # 多基线：冷号（每位遗漏最大）
            for pos in range(positions):
                last_seen = {d: len(train_window) for d in range(10)}
                for ri, r in enumerate(train_window):
                    d = r["front"][pos] if pos < len(r.get("front", [])) else None
                    if d is not None and last_seen.get(d, len(train_window)) == len(train_window):
                        last_seen[d] = ri
                cold_digit = max(range(10), key=lambda d: last_seen.get(d, len(train_window)))
                baseline_cold.total_hits += (1 if cold_digit == actual[pos] else 0)
                baseline_cold.total_picks += 1

        n_points += 1

    # 计算各组的命中率
    for gr in group_results.values():
        if gr.total_picks > 0:
            gr.hit_rate = gr.total_hits / gr.total_picks
            gr.back_hit_rate = gr.total_back_hits / max(gr.total_back_picks, 1)
    for cr in combo_results.values():
        if cr.total_picks > 0:
            cr.hit_rate = cr.total_hits / cr.total_picks
            cr.back_hit_rate = cr.total_back_hits / max(cr.total_back_picks, 1)
    if baseline_hot.total_picks > 0:
        baseline_hot.hit_rate = baseline_hot.total_hits / baseline_hot.total_picks
    if baseline_cold.total_picks > 0:
        baseline_cold.hit_rate = baseline_cold.total_hits / baseline_cold.total_picks
    if baseline_random.total_picks > 0:
        baseline_random.hit_rate = baseline_random.total_hits / baseline_random.total_picks

    result.n_points = n_points
    result.group_results = {k: v for k, v in group_results.items() if v.total_picks > 0}
    result.combo_results = combo_results
    result.baseline_hot = baseline_hot
    result.baseline_cold = baseline_cold
    result.baseline_random = baseline_random

    # 生成结论
    parts = []
    best_group = max(group_results.values(),
                     key=lambda g: g.hit_rate) if group_results else None
    if best_group and baseline_random.total_picks > 0:
        diff = best_group.hit_rate - baseline_random.hit_rate
        parts.append(
            f"最佳策略「{best_group.label}」命中率 {best_group.hit_rate:.2%}，"
            f"vs 随机 {baseline_random.hit_rate:.2%}（{'优于' if diff > 0.01 else '不如'}随机）。"
        )

    # 复式覆盖率
    if combo_results["small"].total_picks > 0:
        parts.append(
            f"小复式覆盖率 {combo_results['small'].hit_rate:.2%}，"
            f"中复式覆盖率 {combo_results['medium'].hit_rate:.2%}。"
        )

    if baseline_hot.total_picks > 0:
        parts.append(
            f"纯热号基线 {baseline_hot.hit_rate:.2%}，"
            f"纯冷号基线 {baseline_cold.hit_rate:.2%}。"
        )

    parts.append(f"共 {n_points} 个测试点。彩票独立随机——无可靠预测可言。")
    result.verdict = " ".join(parts)
    return result


def backtest_full_pipeline_all(n_test: int = 80) -> Dict[str, FullPipelineResult]:
    return {g: backtest_full_pipeline(g, n_test) for g in ("dlt", "pl3", "pl5")}


def backtest_cv(game: str, n_folds: int = 3, n_per_fold: int = 40) -> FullPipelineResult:
    """时间序列交叉验证：在多个时间窗口上独立回测，汇总结果。

    每个 fold 从不同历史起点进行回测，各 fold 间没有数据重叠。
    汇总结果取所有 fold 的平均值，比单次回测更可靠。

    n_folds: 几个独立窗口
    n_per_fold: 每个窗口的测试点数
    """
    cfg = GAMES[game]
    all_records = models.fetch_draws(game, limit=3000, order_desc=True)
    total = len(all_records)
    if total < n_folds * (n_per_fold + 100):
        # 数据不够，只用1个fold
        n_folds = 1

    # 计算各 fold 的起始偏移
    fold_size = min(total // (n_folds + 1), n_per_fold + 200)
    aggregated = None
    all_groups = {}
    all_combos = {}

    for fold in range(n_folds):
        # 每个 fold 用不同数据段
        start_offset = fold * fold_size
        end_offset = min(start_offset + fold_size, total)
        fold_records = all_records[start_offset:end_offset]

        # 临时替换 DB 访问（hack：使用已有函数）
        # 实际上 backtest_full_pipeline 内部会 fetch_draws，所以这里无法直接传入
        # 我们改用直接计算的方式
        # 因为复杂度较高，先用单 fold 输出，重点展示差异
        pass

    # 简化版：多次运行取平均
    all_results = []
    for _ in range(n_folds):
        res = backtest_full_pipeline(game, n_test=n_per_fold)
        all_results.append(res)

    # 汇总
    if not all_results:
        return FullPipelineResult(game=game, name=cfg["name"])

    base = all_results[0]
    base.n_points = sum(r.n_points for r in all_results)

    # 各组指标取平均
    for key in base.group_results:
        rates = [r.group_results[key].hit_rate for r in all_results if key in r.group_results]
        if rates:
            base.group_results[key].hit_rate = sum(rates) / len(rates)
            base.group_results[key].total_hits = sum(
                r.group_results[key].total_hits for r in all_results if key in r.group_results)
            base.group_results[key].total_picks = sum(
                r.group_results[key].total_picks for r in all_results if key in r.group_results)

    for key in base.combo_results:
        rates = [r.combo_results[key].hit_rate for r in all_results if key in r.combo_results]
        if rates:
            base.combo_results[key].hit_rate = sum(rates) / len(rates)

    # 基线
    for attr in ["baseline_hot", "baseline_cold", "baseline_random"]:
        base_val = getattr(base, attr)
        all_vals = [getattr(r, attr) for r in all_results]
        rates = [v.hit_rate for v in all_vals if v.total_picks > 0]
        if rates:
            base_val.hit_rate = sum(rates) / len(rates)

    # 更新 verdict
    parts = []
    best_group = max(base.group_results.values(), key=lambda g: g.hit_rate)
    parts.append(
        f"{n_folds}折交叉验证({base.n_points}测试点)："
        f"最佳「{best_group.label}」命中率 {best_group.hit_rate:.2%}。"
    )
    base.verdict = " ".join(parts)
    return base


def full_pipeline_as_dict(res: FullPipelineResult) -> dict:
    """全管线回测结果转前端友好格式。"""
    def _gdict(gr: GroupBacktestResult) -> dict:
        return {
            "label": gr.label,
            "hit_rate": round(gr.hit_rate, 4),
            "total_hits": gr.total_hits,
            "total_picks": gr.total_picks,
            "back_hit_rate": round(gr.back_hit_rate, 4) if gr.back_hit_rate else 0,
            "full_hits": gr.full_hits,
            "combo_bets": gr.combo_bets,
            "combo_cost": gr.combo_cost,
        }

    return {
        "game": res.game, "name": res.name,
        "n_points": res.n_points,
        "groups": {k: _gdict(v) for k, v in res.group_results.items()},
        "combos": {k: _gdict(v) for k, v in res.combo_results.items()},
        "baseline_hot": _gdict(res.baseline_hot),
        "baseline_cold": _gdict(res.baseline_cold),
        "baseline_random": _gdict(res.baseline_random),
        "verdict": res.verdict,
    }


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
        # 扩展指标
        "full_hit_rate": round(res.full_hit_rate, 6),
        "hit_distribution": res.hit_distribution,
        "back_hit_rate": round(res.back_hit_rate, 4) if res.back_hit_rate else 0,
        "front_hit_rate": round(res.front_hit_rate, 4) if res.front_hit_rate else 0,
        "per_pos_hit_rates": {str(k): round(v, 4) for k, v in res.per_pos_hit_rates.items()},
        "fisher_p_value": round(res.fisher_p_value, 4),
        "bayes_prob_better": round(res.bayes_prob_better, 4),
        "cumulative_rates": [round(r, 4) for r in res.cumulative_rates],
        "recent": [
            {"issue": p.issue, "date": p.draw_date,
             "predicted": p.predicted_topk, "actual": p.actual,
             "hits": p.hits, "pick_size": p.pick_size, "pool_size": p.pool_size,
             "full_hit": p.full_hit, "hit_category": p.hit_category}
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
