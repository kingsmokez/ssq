# -*- coding: utf-8 -*-
"""特征调参自动优化 — 网格搜索评分权重和半衰期。

设计动机：当前评分公式的权重 (freq=0.45, omission=0.30, momentum=0.25)
和 EWMA 半衰期 (25) 是手动设定的。本脚本在回测循环中搜索最优参数组合。

⚠️ 重要：彩票是独立随机事件，最优参数 ≈ 任意参数。
   本脚本的价值是"诚实证明调参也提高不了命中率"，而非真正找到"更准的公式"。
   如果搜索结果显示所有组合的命中率都在 ±5% 噪声带内，这就是期望结果。
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lottery import models
from lottery.analyzer import _build_window, _per_number_stats
from lottery.backtester import backtest, BacktestResult
from config import GAMES, STATS_WINDOW_DAYS


# ---------- 可调参数空间 ----------

WEIGHT_GRID = [
    {"freq": 0.60, "omission": 0.20, "momentum": 0.20},  # 偏重频率
    {"freq": 0.50, "omission": 0.30, "momentum": 0.20},  # 均衡偏频率
    {"freq": 0.45, "omission": 0.30, "momentum": 0.25},  # 当前配置
    {"freq": 0.40, "omission": 0.35, "momentum": 0.25},  # 偏重遗漏
    {"freq": 0.33, "omission": 0.33, "momentum": 0.34},  # 完全均衡
    {"freq": 0.30, "omission": 0.40, "momentum": 0.30},  # 偏重遗漏
    {"freq": 0.25, "omission": 0.50, "momentum": 0.25},  # 大幅偏重遗漏
    {"freq": 0.50, "omission": 0.25, "momentum": 0.25},  # 偏热
    {"freq": 0.20, "omission": 0.30, "momentum": 0.50},  # 偏重趋势
    {"freq": 0.70, "omission": 0.15, "momentum": 0.15},  # 极度偏热
]

HALF_LIFE_GRID = [10, 15, 20, 25, 30, 40, 50, 75, 100]

MOMENTUM_N_GRID = [15, 20, 25, 30, 40, 50]

COLD_HOT_N_GRID = [30, 40, 50, 60, 80]


def _patch_weights(new_weights: dict, half_life: int,
                   momentum_n: int, cold_hot_n: int) -> tuple:
    """临时修改 config 模块中的参数，返回 (旧权重, 旧半衰期, 旧MomentumN, 旧ColdHotN)。"""
    import config
    old_w = dict(config.WEIGHTS)
    old_hl = config.MOMENTUM_RECENT_N  # 实际是 momentum_n 的同义词
    old_mn = config.MOMENTUM_RECENT_N
    old_cn = config.COLD_HOT_RECENT_N
    config.WEIGHTS.update(new_weights)
    # 注意：EWMA half_life 在 features.py 中是硬编码的，需要通过 analyzer.py 传入
    # 但当前代码 half_life=25 是硬编码在 _per_number_stats 里调用的 ewma_freq(presence, half_life=25)
    # 无法直接从 config 修改。我们记录这个限制。
    config.MOMENTUM_RECENT_N = momentum_n
    config.COLD_HOT_RECENT_N = cold_hot_n
    return old_w, old_hl, old_mn, old_cn


def _restore_config(old_w, old_hl, old_mn, old_cn):
    """恢复原有配置。"""
    import config
    config.WEIGHTS.clear()
    config.WEIGHTS.update(old_w)
    config.MOMENTUM_RECENT_N = old_mn  # 用旧值恢复两个参数
    config.COLD_HOT_RECENT_N = old_cn


def search(game: str = "dlt", n_test: int = 60,
           verbose: bool = True) -> list:
    """网格搜索最优参数组合。

    返回按命中率降序排列的 [(params, hit_rate, lift), ...]
    """
    print(f"开始网格搜索 [{game}] ...")
    print(f"  权重组合: {len(WEIGHT_GRID)}")
    print(f"  半衰期: {len(HALF_LIFE_GRID)} (当前硬编码在 features.py，仅用于记录)")
    print(f"  Momentum N: {len(MOMENTUM_N_GRID)}")
    print(f"  ColdHot N: {len(COLD_HOT_N_GRID)}")
    print(f"  总组合数: {len(WEIGHT_GRID) * len(MOMENTUM_N_GRID) * len(COLD_HOT_N_GRID)}")
    print(f"  每个组合 {n_test} 个测试点")
    print(f"  ⚠️ EWMA half_life 在 features.py 硬编码为 25，本次不搜索\n")

    results = []
    total = len(WEIGHT_GRID) * len(MOMENTUM_N_GRID) * len(COLD_HOT_N_GRID)
    count = 0

    for w in WEIGHT_GRID:
        for mn in MOMENTUM_N_GRID:
            for cn in COLD_HOT_N_GRID:
                count += 1
                old = _patch_weights(w, 25, mn, cn)
                try:
                    res = backtest(game, n_test=n_test)
                    results.append({
                        "weights": dict(w),
                        "momentum_n": mn,
                        "cold_hot_n": cn,
                        "model_hit_rate": res.model_hit_rate,
                        "random_hit_rate": res.random_hit_rate,
                        "lift": res.lift,
                        "p_value": res.p_value,
                        "full_hit_rate": res.full_hit_rate,
                        "bayes_prob_better": res.bayes_prob_better,
                    })
                    if verbose:
                        print(f"  [{count}/{total}] w={w} mn={mn} cn={cn} → "
                              f"命中率={res.model_hit_rate:.2%}, "
                              f"提升={res.lift:+.2%}, "
                              f"贝叶斯P={res.bayes_prob_better:.1%}")
                finally:
                    _restore_config(*old)

    # 按命中率降序排序
    results.sort(key=lambda x: -x["model_hit_rate"])
    return results


def print_summary(results: list, top_n: int = 5):
    """打印搜索摘要。"""
    print(f"\n{'='*70}")
    print(f"网格搜索完成，共 {len(results)} 个组合")
    print(f"{'='*70}")

    # 检查是否有实际差异
    rates = [r["model_hit_rate"] for r in results]
    lifts = [r["lift"] for r in results]
    max_rate = max(rates)
    min_rate = min(rates)
    spread = max_rate - min_rate

    print(f"\n命中率范围: {min_rate:.2%} ~ {max_rate:.2%} (差距 {spread:.2%})")
    print(f"提升范围: {min(lifts):+.2%} ~ {max(lifts):+.2%}")

    if spread < 0.05:
        print(f"\n⚠️ 所有参数组合的命中率差距 < 5%，说明参数选择对结果影响极小。")
        print(f"   这是彩票独立随机本质的数学体现——没有参数能真正『提高』命中率。")
        print(f"   当前默认配置 (freq=0.45, omission=0.30, momentum=0.25) 与其他参数无本质差别。")
    else:
        print(f"\n⚠️ 参数间存在 >5% 的差距，但需要考虑这是否是噪声（每个组合的测试点有限）。")
        print(f"   建议增加 n_test 重新搜索确认。")

    print(f"\n--- Top {top_n} 参数组合 ---")
    for i, r in enumerate(results[:top_n]):
        print(f"  #{i+1}: 权重={r['weights']}, momentum_n={r['momentum_n']}, "
              f"cold_hot_n={r['cold_hot_n']}")
        print(f"       命中率={r['model_hit_rate']:.2%}, 随机基线={r['random_hit_rate']:.2%}, "
              f"提升={r['lift']:+.2%}")
        print(f"       p值={r['p_value']:.4f}, 贝叶斯P(优于随机)={r['bayes_prob_better']:.1%}")

    print(f"\n实际生产中建议使用当前默认配置（或随机——反正都一样）。")


if __name__ == "__main__":
    models.init_db()
    game = sys.argv[1] if len(sys.argv) > 1 else "dlt"
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    results = search(game, n_test=n_test)
    print_summary(results)
