# -*- coding: utf-8 -*-
"""统计引擎 — 基于历史开奖计算多维特征。

输入：从 DB 取某玩法的历史开奖（最新在前）。
输出：一个 GameStats 对象，包含：
  - lotto: front_stats/back_stats，每个号码 {freq, omission, avg_omission, omission_ratio, momentum, cold_hot}
  - digit: position_stats[pos][digit] 同上结构
  - 组合约束（lotto）：和值/奇偶/大小/跨度的历史分布区间

所有特征基于"近 STATS_WINDOW_DAYS"窗口。
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List

from config import (
    GAMES, STATS_WINDOW_DAYS, MOMENTUM_RECENT_N, COLD_HOT_RECENT_N,
    BIG_SMALL_THRESHOLD,
)
from . import models


@dataclass
class NumberStat:
    """单个候选号码的统计特征。"""
    number: int
    freq: float = 0.0              # 出现频率（窗口内出现期数/总期数）
    omission: int = 0              # 当前遗漏（距上次开出的期数）
    avg_omission: float = 0.0      # 平均遗漏（窗口内）
    omission_ratio: float = 0.0    # 当前遗漏 / 平均遗漏
    momentum: float = 0.0          # 近 N 期频率 - 全窗口频率（>0 升温）
    last_seen: int = 0             # 距今多少期开出（0=最近一期开出）
    ewma_freq: float = 0.0         # 指数加权频率（近期权重更高，更平滑）
    bayes_freq: float = 0.0        # 贝叶斯平滑频率（抗小样本噪声）
    omission_pct: float = 0.5      # 当前遗漏在历史分布中的百分位 [0,1]
    cold_hot: str = "平"           # 冷/热/平 标签
    warm_tendency: str = "均衡"    # 频率/遗漏倾向: "偏热"(频率主导) / "偏冷"(遗漏主导) / "均衡"


@dataclass
class GameStats:
    """某玩法的完整统计快照。"""
    game: str
    name: str
    total_draws: int = 0
    window_draws: int = 0
    # lotto 玩法
    front_stats: Dict[int, NumberStat] = field(default_factory=dict)
    back_stats: Dict[int, NumberStat] = field(default_factory=dict)
    # digit 玩法：position -> digit -> stat
    position_stats: Dict[int, Dict[int, NumberStat]] = field(default_factory=dict)
    # lotto 组合约束
    sum_range: tuple = (0, 0)          # 前区和值众数区间
    odd_ratio_common: str = ""         # 奇偶比主流（如 "3:2"）
    big_small_ratio_common: str = ""   # 大小比主流
    span_range: tuple = (0, 0)         # 跨度众数区间
    cooccur: Dict = field(default_factory=dict)  # 前区号码共生矩阵 {n: {m: rate}}
    # digit 组合约束
    digit_sum_range: tuple = (0, 0)    # 各位和值众数区间
    digit_odd_count_common: int = 0    # 最常见奇数个数
    latest_numbers: List[int] = field(default_factory=list)
    latest_issue: str = ""
    latest_date: str = ""
    computed_at: str = ""


def _build_window(records, days: int):
    """按日期窗口过滤记录。records 最新在前。返回窗口内记录（最新在前）。
    若记录缺日期，退化为按数量取近 N 条。"""
    if not records:
        return []
    cutoff = ""
    # 找最新有效日期
    ref = None
    for r in records:
        d = r["draw_date"]
        if d:
            try:
                ref = datetime.strptime(d[:10], "%Y-%m-%d")
                break
            except ValueError:
                continue
    if ref is None:
        # 无日期，按数量退化
        return records[: min(len(records), 1500)]
    cutoff_dt = ref - timedelta(days=days)
    cutoff = cutoff_dt.strftime("%Y-%m-%d")
    window = [r for r in records if r["draw_date"] and r["draw_date"][:10] >= cutoff]
    return window if window else records


def _per_number_stats(pool: List[int], seq_of_sets, total: int, momentum_n: int,
                      cold_n: int) -> Dict[int, NumberStat]:
    """通用：对一组候选号（pool）计算各特征。
    seq_of_sets: 每期开奖的号码集合列表，最新在前（index 0 最新）。
    """
    stats = {n: NumberStat(number=n) for n in pool}
    if total == 0 or not seq_of_sets:
        return stats

    # 频率 & 遗漏 & last_seen
    for n in pool:
        # 记录 n 在窗口内出现的位置索引（index 0 = 最新期）
        appear_indices = []
        for i, s in enumerate(seq_of_sets):
            if n in s:
                appear_indices.append(i)

        st = stats[n]
        appear_count = len(appear_indices)
        st.freq = appear_count / total
        st.last_seen = appear_indices[0] if appear_indices else total
        st.omission = st.last_seen

        # 真实平均遗漏：已完成遗漏段（两次出现之间的间隔）的算术平均
        # 不含当前进行中的遗漏段，因为当前遗漏是"未完成"的
        if appear_count >= 2:
            gaps = [appear_indices[i + 1] - appear_indices[i]
                    for i in range(len(appear_indices) - 1)]
            st.avg_omission = sum(gaps) / len(gaps)
        elif appear_count == 1:
            # 只出现1次，无法计算遗漏段均值，退化用窗口期数
            st.avg_omission = total
        else:
            st.avg_omission = total
        st.omission_ratio = (st.omission / st.avg_omission) if st.avg_omission > 0 else 0.0

        # 增强特征：贝叶斯平滑频率（抗小样本噪声）
        from .features import bayes_smoothed_freq, ewma_freq, omission_percentile
        st.bayes_freq = bayes_smoothed_freq(appear_count, total, len(pool))
        # 指数加权频率：从 seq_of_sets 提取该号的出现序列（最新在前）
        presence = [1 if n in s else 0 for s in seq_of_sets]
        st.ewma_freq = ewma_freq(presence, half_life=25)
        # 遗漏分位
        if appear_count >= 2:
            gaps = [appear_indices[i + 1] - appear_indices[i]
                    for i in range(len(appear_indices) - 1)]
            st.omission_pct = omission_percentile(st.omission, gaps)
        else:
            st.omission_pct = 0.5

    # momentum：近 momentum_n 期频率 - 全窗口频率
    recent = seq_of_sets[:momentum_n]
    recent_total = len(recent)
    for n in pool:
        recent_count = sum(1 for s in recent if n in s)
        recent_freq = recent_count / recent_total if recent_total else 0
        stats[n].momentum = recent_freq - stats[n].freq

    # cold/hot 标签：近 cold_n 期
    recent_cold = seq_of_sets[:cold_n]
    cold_total = len(recent_cold)
    for n in pool:
        cnt = sum(1 for s in recent_cold if n in s)
        rf = cnt / cold_total if cold_total else 0
        expect = 1.0 / len(pool) if pool else 0
        if rf >= expect * 1.3:
            stats[n].cold_hot = "热"
        elif rf <= expect * 0.6:
            stats[n].cold_hot = "冷"
        else:
            stats[n].cold_hot = "平"

    # warm_tendency：标记号码的频率/遗漏主导倾向
    # freq 和 omission_ratio 在评分中是互补信号：高频=热，高遗漏比=冷
    # warm_tendency 帮助理解每个号码的实际倾向
    expect_freq = 1.0 / len(pool) if pool else 0
    for n in pool:
        st = stats[n]
        if st.freq > expect_freq * 1.2 and st.omission_ratio < 0.8:
            st.warm_tendency = "偏热"     # 高频+低遗漏比 → 频率主导
        elif st.omission_ratio > 1.5 and st.freq < expect_freq * 0.8:
            st.warm_tendency = "偏冷"     # 低频+高遗漏比 → 遗漏主导
        else:
            st.warm_tendency = "均衡"     # 均衡或矛盾信号
    return stats


def _combo_constraints(front_sets):
    """大乐透前区组合约束（基于窗口）。front_sets: 每期前区号码列表，最新在前。"""
    if not front_sets:
        return (0, 0), "", "", (0, 0)
    sums = []
    odd_ratios = []
    bs_ratios = []
    spans = []
    for fs in front_sets:
        if len(fs) < 2:
            continue
        sums.append(sum(fs))
        odd = sum(1 for x in fs if x % 2 == 1)
        odd_ratios.append(odd)
        big = sum(1 for x in fs if x >= BIG_SMALL_THRESHOLD)
        bs_ratios.append(big)
        spans.append(max(fs) - min(fs))
    if not sums:
        return (0, 0), "", "", (0, 0)
    sums.sort()
    spans.sort()
    # 取中间 60% 作为"众数区间"
    lo = int(len(sums) * 0.2)
    hi = int(len(sums) * 0.8)
    sum_range = (sums[lo], sums[hi]) if hi > lo else (sums[0], sums[-1])
    slo = int(len(spans) * 0.2)
    shi = int(len(spans) * 0.8)
    span_range = (spans[slo], spans[shi]) if shi > slo else (spans[0], spans[-1])
    # 最常见奇偶比/大小比
    from collections import Counter
    odd_cnt = Counter(odd_ratios).most_common(1)[0][0]
    bs_cnt = Counter(bs_ratios).most_common(1)[0][0]
    odd_ratio_str = f"{odd_cnt}:{len(sums) and (5 - odd_cnt)}"
    bs_ratio_str = f"{bs_cnt}:{5 - bs_cnt}"
    return sum_range, odd_ratio_str, bs_ratio_str, span_range


def _digit_combo_constraints(window_records, positions):
    """计算 digit 玩法的组合约束（和值区间、奇数个数众数）。

    window_records: 窗口内开奖记录，最新在前。
    positions: 位数（排列3=3, 排列5=5）。
    """
    sums = []
    odd_counts = []
    for r in window_records:
        digits = r["front"]
        if len(digits) < positions:
            continue
        sums.append(sum(digits))
        odd_counts.append(sum(1 for d in digits if d % 2 == 1))
    if not sums:
        return (0, 0), 0
    sums.sort()
    lo = int(len(sums) * 0.2)
    hi = int(len(sums) * 0.8)
    sum_range = (sums[lo], sums[hi]) if hi > lo else (sums[0], sums[-1])
    from collections import Counter
    odd_common = Counter(odd_counts).most_common(1)[0][0]
    return sum_range, odd_common


def analyze(game: str, days: int = STATS_WINDOW_DAYS) -> GameStats:
    """计算某玩法统计快照。"""
    cfg = GAMES[game]
    records = models.fetch_draws(game, limit=5000, order_desc=True)
    gs = GameStats(
        game=game, name=cfg["name"], total_draws=len(records),
        latest_numbers=records[0]["front"] + records[0]["back"] if records else [],
        latest_issue=records[0]["issue"] if records else "",
        latest_date=records[0]["draw_date"][:10] if records and records[0]["draw_date"] else "",
        computed_at=datetime.now().isoformat(timespec="seconds"),
    )
    if not records:
        return gs

    window = _build_window(records, days)
    gs.window_draws = len(window)

    if cfg["type"] == "lotto":
        front_sets = [set(r["front"]) for r in window]
        back_sets = [set(r["back"]) for r in window]
        gs.front_stats = _per_number_stats(
            cfg["front_pool"], front_sets, len(window),
            MOMENTUM_RECENT_N, COLD_HOT_RECENT_N)
        gs.back_stats = _per_number_stats(
            cfg["back_pool"], back_sets, len(window),
            MOMENTUM_RECENT_N, COLD_HOT_RECENT_N)
        gs.sum_range, gs.odd_ratio_common, gs.big_small_ratio_common, gs.span_range = \
            _combo_constraints([r["front"] for r in window])
        # 计算前区号码共生矩阵
        from .features import cooccurrence_matrix
        gs.cooccur = cooccurrence_matrix(
            [r["front"] for r in window], cfg["front_pool"])
    else:
        positions = cfg["positions"]
        pool = cfg["digit_pool"]
        gs.position_stats = {}
        for pos in range(positions):
            # 每个位置该位的数字集合（每期 1 个）
            seq = []
            for r in window:
                digits = r["front"]
                if pos < len(digits):
                    seq.append({digits[pos]})
                else:
                    seq.append(set())
            gs.position_stats[pos] = _per_number_stats(
                pool, seq, len(window), MOMENTUM_RECENT_N, COLD_HOT_RECENT_N)
        # digit 组合约束
        gs.digit_sum_range, gs.digit_odd_count_common = \
            _digit_combo_constraints(window, positions)
    return gs


def analyze_all(days: int = STATS_WINDOW_DAYS) -> Dict[str, GameStats]:
    return {g: analyze(g, days) for g in ("dlt", "pl3", "pl5")}


if __name__ == "__main__":
    models.init_db()
    for g in ("dlt", "pl3", "pl5"):
        gs = analyze(g)
        print(f"\n===== {gs.name} ({g}) =====")
        print(f"  总期数 {gs.total_draws}，窗口期数 {gs.window_draws}，最新 {gs.latest_issue} ({gs.latest_date})")
        if gs.front_stats:
            top = sorted(gs.front_stats.values(), key=lambda s: -s.freq)[:5]
            print("  前区频率 Top5:", [(s.number, round(s.freq, 3), f"遗漏{s.omission}", s.cold_hot) for s in top])
            cold = sorted(gs.back_stats.values(), key=lambda s: -s.omission_ratio)[:3]
            print("  后区高遗漏比 Top3:", [(s.number, round(s.omission_ratio, 2), f"遗漏{s.omission}") for s in cold])
            print(f"  组合: 和值{gs.sum_range} 奇偶{gs.odd_ratio_common} 大小{gs.big_small_ratio_common} 跨度{gs.span_range}")
        else:
            for pos in sorted(gs.position_stats):
                stats = gs.position_stats[pos]
                top = sorted(stats.values(), key=lambda s: -s.freq)[:3]
                print(f"  第{pos+1}位 Top3:", [(s.number, round(s.freq, 3), s.cold_hot) for s in top])
