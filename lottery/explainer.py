# -*- coding: utf-8 -*-
"""理由生成器 — 为每组推荐生成人类可读的中文理由，引用具体统计数据。"""

from typing import List
from .analyzer import GameStats


def _fmt_nums(nums: List[int]) -> str:
    return "、".join(str(n).zfill(2) if n < 10 else str(n) for n in nums)


def _stat_for(gs: GameStats, number: int, zone: str = "front"):
    pool = gs.front_stats if zone == "front" else gs.back_stats
    return pool.get(number)


def _hot_trend_words(stats_pool) -> str:
    """从号池里挑 1-2 个有代表性的趋势词。"""
    if not stats_pool:
        return ""
    rising = sorted(stats_pool.values(), key=lambda s: -s.momentum)[:1]
    cold = sorted(stats_pool.values(), key=lambda s: -s.omission_ratio)[:1]
    parts = []
    if rising and rising[0].momentum > 0:
        s = rising[0]
        tendency = {"偏热": "升温", "偏冷": "降温", "均衡": "中性"}.get(s.warm_tendency, "")
        parts.append(f"{s.number:02d} 近期{tendency}(动量+{s.momentum:.3f})")
    if cold and cold[0].omission_ratio > 1.2:
        s = cold[0]
        tendency = {"偏热": "偏热", "偏冷": "偏冷", "均衡": "中性"}.get(s.warm_tendency, "")
        parts.append(f"{s.number:02d} 已遗漏{s.omission}期(均{s.avg_omission:.0f}期，{tendency})")
    return "；".join(parts)


def explain_group(gs: GameStats, picks: dict, label: str, strategy: str) -> str:
    """为单选组生成理由。"""
    if "front" in picks:  # lotto
        front, back = picks["front"], picks["back"]
        front_sum = sum(front)
        odd = sum(1 for x in front if x % 2 == 1)
        big = sum(1 for x in front if x >= 18)
        reason = f"【{label}】"
        if strategy == "balanced":
            reason += f"综合频率/遗漏/趋势打分最高。前区 {_fmt_nums(front)} 在窗口内出现频率领先，"
        elif strategy == "cold":
            reason += f"前区 {_fmt_nums(front)} 侧重遗漏比偏高的冷号（近期该出），"
        elif strategy == "momentum":
            reason += f"前区 {_fmt_nums(front)} 近30期出现率明显上升（追升温），"
        elif strategy == "random":
            reason += f"前区 {_fmt_nums(front)} 在高分号池中随机组合（幸运组），"
        else:  # 组合合规
            # 计算组合内共生率信息
            cooc = gs.cooccur
            cooc_hints = []
            for i, a in enumerate(front):
                for b in front[i + 1:]:
                    rate = cooc.get(a, {}).get(b, 0)
                    if rate > 0:
                        cooc_hints.append((a, b, rate))
            cooc_text = ""
            if cooc_hints:
                # 取共生率最高的3对
                top3 = sorted(cooc_hints, key=lambda x: -x[2])[:3]
                pairs_str = "、".join(f"{a:02d}与{b:02d}({r:.1%})" for a, b, r in top3)
                avg_rate = sum(r for _, _, r in cooc_hints) / len(cooc_hints)
                cooc_text = f"号码间历史共生率较高(均值{avg_rate:.1%}，如{pairs_str})，"
            reason += f"前区 {_fmt_nums(front)}，{cooc_text}"
        trend = _hot_trend_words({n: _stat_for(gs, n) for n in front if _stat_for(gs, n)})
        if trend:
            reason += f"其中{trend}。"
        reason += (f"本组和值 {front_sum} 落在历史高频区间 [{gs.sum_range[0]},{gs.sum_range[1]}]，"
                   f"奇偶比 {odd}:{5-odd}，大小比 {big}:{5-big}（主流为 {gs.odd_ratio_common}/"
                   f"{gs.big_small_ratio_common}）。后区 {_fmt_nums(back)} 评分靠前。")
        return reason
    else:  # digit
        digits = picks["digits"]
        reason = f"【{label}】各位取评分最高数字："
        bits = []
        for pos, d in enumerate(digits):
            st = gs.position_stats[pos].get(d)
            if st:
                bits.append(f"第{pos+1}位={d}(频率{st.freq:.1%},{st.cold_hot})")
        reason += "，".join(bits) + "。"
        if label == "组合合规":
            digit_sum = sum(digits)
            odd_count = sum(1 for x in digits if x % 2 == 1)
            reason += (f"本组和值 {digit_sum} 落在历史高频区间 [{gs.digit_sum_range[0]},{gs.digit_sum_range[1]}]，"
                       f"奇数个数 {odd_count}（主流为 {gs.digit_odd_count_common}个）。")
        elif strategy == "cold":
            reason += "本组侧重各位置遗漏比偏高的数字（追冷回补）。"
        elif strategy == "momentum":
            reason += "本组侧重各位置近期升温的数字。"
        elif strategy == "diverse":
            reason += "本组在高分池中侧重与前组不同的号码组合。"
        return reason


def explain_combo(gs: GameStats, picks: dict, mode: str, bets: int) -> str:
    """为复式生成理由。"""
    size_cn = {"small": "小复式", "medium": "中复式", "single": "单注"}.get(mode, mode)
    if "front" in picks:  # lotto
        front, back = picks["front"], picks["back"]
        reason = (f"【{size_cn}】前区取 {_fmt_nums(front)}（共{len(front)}个），"
                  f"后区取 {_fmt_nums(back)}（共{len(back)}个），"
                  f"合计 {bets} 注。覆盖高频+冷热兼顾的号码池，"
                  f"前区和值区间建议 {gs.sum_range[0]}-{gs.sum_range[1]}。")
        return reason
    else:  # digit
        grid = picks["digits_grid"]
        reason = f"【{size_cn}】每位候选：" + "  ".join(
            f"第{i+1}位:[{'/'.join(map(str, g))}]" for i, g in enumerate(grid)
        ) + f"，合计 {bets} 注组合。每位的候选均为该位置综合评分靠前的数字。"
        return reason
