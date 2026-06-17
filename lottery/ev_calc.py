# -*- coding: utf-8 -*-
"""期望值与覆盖率计算器 — 本系统唯一有数学意义的"优化"。

设计动机：消融实验已证明任何特征工程的命中率 ≈ 纯随机（lift 落在 ±10% 噪声带）。
因此对用户真正有价值的不是"更准的号"，而是诚实地回答两个问题：

1. 给定一个复式方案，"至少中一位"的概率是多少？花这笔钱值不值？
2. 给定预算，应该买什么档位的复式？

所有函数为纯函数，便于回测/测试。

⚠️ 关键数学事实（独立同分布假设）：
   - digit 每位命中概率 p = 候选数 / 10
   - "至少中一位" = 1 - ∏(1 - p_i)
   - "直选中奖"（digit 全对）= ∏ p_i
   - lotto 组选：超几何分布（无放回抽样）
"""

from dataclasses import dataclass
from typing import List
from functools import reduce

from config import GAMES, BET_UNIT_PRICE


# ---------------- digit（排列3/5）----------------

@dataclass
class DigitCoverage:
    """digit 复式方案的覆盖率分析。"""
    candidates_per_pos: List[int]   # 每位候选个数，如 [2,2,2]
    bets: int                       = 1
    cost: int                       = 0
    p_at_least_one: float           = 0.0   # 至少中一位概率
    p_all_hit: float                = 0.0   # 直选中奖概率（位置全对）
    p_none: float                   = 0.0   # 全不中概率


def digit_coverage(candidates_per_pos: List[int], unit_price: int = 2) -> DigitCoverage:
    """计算 digit 复式方案的覆盖情况。

    candidates_per_pos: 每位选了几个号（排列3=3个元素，排列5=5个元素）。
    每位号池固定为 0-9（10 个），命中概率 = 候选数/10。
    """
    bets = 1
    for c in candidates_per_pos:
        bets *= max(1, c)
    cost = bets * unit_price
    # 每位命中概率
    p_per = [c / 10.0 for c in candidates_per_pos]
    # 全不中 = ∏(1-p)
    p_none = reduce(lambda acc, p: acc * (1 - p), p_per, 1.0)
    p_at_least_one = 1 - p_none
    # 直选中奖 = ∏p
    p_all = reduce(lambda acc, p: acc * p, p_per, 1.0)
    return DigitCoverage(
        candidates_per_pos=candidates_per_pos, bets=bets, cost=cost,
        p_at_least_one=p_at_least_one, p_all_hit=p_all, p_none=p_none,
    )


def digit_single_prob(positions: int = 3) -> dict:
    """digit 单注（每位 1 个号）的概率提示。"""
    # 每位命中 1/10
    p_none = (9 / 10) ** positions
    return {
        "p_none": p_none,
        "p_at_least_one": 1 - p_none,
        "p_all_hit": 1 / (10 ** positions),   # 直选中奖
        "positions": positions,
    }


# ---------------- lotto（大乐透）----------------

from math import comb

@dataclass
class LottoCoverage:
    """lotto 复式方案的覆盖率分析。"""
    front_pick: int     # 前区选几个
    back_pick: int      # 后区选几个
    front_pool: int     # 前区号池（35）
    back_pool: int      # 后区号池（12）
    draw_front: int     # 每期开几个前区（5）
    draw_back: int      # 每期开几个后区（2）
    bets: int           = 1
    cost: int           = 0
    p_front_at_least_one: float  = 0.0   # 前区至少中 1 个
    p_back_at_least_one: float   = 0.0   # 后区至少中 1 个
    p_any_hit: float             = 0.0   # 前或后至少中 1 个
    p_first_prize: float         = 0.0   # 一等奖（前5后2全中）概率


def lotto_coverage(front_pick: int, back_pick: int,
                   unit_price: int = 2,
                   front_pool: int = 35, back_pool: int = 12,
                   draw_front: int = 5, draw_back: int = 2) -> LottoCoverage:
    """大乐透复式覆盖率（超几何分布，无放回）。

    p(前区至少中1) = 1 - C(35-5, front_pick) / C(35, front_pick)
      含义：从 35 选 front_pick 个，避开全部 5 个开奖号的概率。
    """
    bets = comb(front_pick, draw_front) * comb(back_pick, draw_back)
    cost = bets * unit_price
    # 前区一个都不中 = 选的 front_pick 个全在 30 个非开奖号里
    p_front_none = comb(front_pool - draw_front, front_pick) / comb(front_pool, front_pick)
    p_front_one = 1 - p_front_none
    # 后区同理
    p_back_none = comb(back_pool - draw_back, back_pick) / comb(back_pool, back_pick)
    p_back_one = 1 - p_back_none
    # 前/后任一中 = 1 - 都不中
    p_any = 1 - p_front_none * p_back_none
    # 一等奖：选的号正好包含全部开奖号
    p_first = (comb(draw_front, draw_front) * comb(front_pool - draw_front, front_pick - draw_front)
               / comb(front_pool, front_pick)) * (
               comb(draw_back, draw_back) * comb(back_pool - draw_back, back_pick - draw_back)
               / comb(back_pool, back_pick))
    return LottoCoverage(
        front_pick=front_pick, back_pick=back_pick,
        front_pool=front_pool, back_pool=back_pool,
        draw_front=draw_front, draw_back=draw_back,
        bets=bets, cost=cost,
        p_front_at_least_one=p_front_one,
        p_back_at_least_one=p_back_one,
        p_any_hit=p_any, p_first_prize=p_first,
    )


# ---------------- 预算建议 ----------------

@dataclass
class BudgetPlan:
    """给定预算，推荐最划算的复式档位。"""
    budget: int
    game: str
    recommended: dict    # 推荐档位信息
    alternatives: List[dict]


def recommend_budget(game: str, budget: int) -> BudgetPlan:
    """给定玩法和预算（元），推荐最划算的复式档位。

    策略：枚举所有候选档位，按成本不超预算排序，选"至少中一位概率/元"最高的。
    """
    cfg = GAMES[game]
    unit = BET_UNIT_PRICE.get(game, 2)
    candidates = []

    if cfg["type"] == "digit":
        positions = cfg["positions"]
        # 候选：每位选 1~4 个
        from itertools import product
        for combo in product(range(1, 5), repeat=positions):
            cov = digit_coverage(list(combo), unit)
            if cov.cost > budget:
                continue
            ratio = cov.p_at_least_one / cov.cost if cov.cost else 0
            candidates.append({
                "candidates_per_pos": list(combo),
                "bets": cov.bets, "cost": cov.cost,
                "p_at_least_one": cov.p_at_least_one,
                "p_all_hit": cov.p_all_hit,
                "coverage_prob": cov.p_at_least_one,
                "value_ratio": ratio,
            })
    else:
        # lotto：前区 5~8，后区 2~4
        for fp in range(5, 9):
            for bp in range(2, 5):
                cov = lotto_coverage(fp, bp, unit,
                                     len(cfg["front_pool"]), len(cfg["back_pool"]),
                                     cfg["front_pick"], cfg["back_pick"])
                if cov.cost > budget:
                    continue
                ratio = cov.p_any_hit / cov.cost if cov.cost else 0
                candidates.append({
                    "front_pick": fp, "back_pick": bp,
                    "bets": cov.bets, "cost": cov.cost,
                    "p_any_hit": cov.p_any_hit,
                    "p_first_prize": cov.p_first_prize,
                    "coverage_prob": cov.p_any_hit,
                    "value_ratio": ratio,
                })

    if not candidates:
        # 预算过低，给最低档
        if cfg["type"] == "digit":
            cov = digit_coverage([1] * cfg["positions"], unit)
            candidates.append({
                "candidates_per_pos": [1] * cfg["positions"],
                "bets": cov.bets, "cost": cov.cost,
                "p_at_least_one": cov.p_at_least_one,
                "p_all_hit": cov.p_all_hit,
                "coverage_prob": cov.p_at_least_one, "value_ratio": 0,
            })
        else:
            cov = lotto_coverage(cfg["front_pick"], cfg["back_pick"], unit,
                                 len(cfg["front_pool"]), len(cfg["back_pool"]),
                                 cfg["front_pick"], cfg["back_pick"])
            candidates.append({
                "front_pick": cfg["front_pick"], "back_pick": cfg["back_pick"],
                "bets": cov.bets, "cost": cov.cost,
                "p_any_hit": cov.p_any_hit,
                "p_first_prize": cov.p_first_prize,
                "coverage_prob": cov.p_any_hit, "value_ratio": 0,
            })

    # 排序策略：
    # 1) 在不超预算的前提下，优先花掉预算提升覆盖率（"至少中一位"概率高的优先）
    # 2) 概率相同时，成本低的优先（更省）
    # 不用 p/cost —— 那会让最便宜的单注永远排第一，失去"花预算提覆盖率"的意义。
    candidates.sort(key=lambda x: (-x["coverage_prob"], x["cost"]))
    best = candidates[0]
    alts = candidates[1:5]
    return BudgetPlan(budget=budget, game=game, recommended=best, alternatives=alts)


# ---------------- 贪心集合覆盖优化 ----------------

@dataclass
class GreedyCoverPlan:
    """贪心集合覆盖的结果。"""
    selected_tickets: List[List[int]]   # 选中的号码组合
    total_bets: int                     # 总注数
    total_cost: int                     # 总成本
    covered_numbers: set                # 覆盖的所有不同号码
    coverage_rate: float                # 覆盖的号码占总号池比例
    redundancy: float                   # 冗余率（号码在多个注中重复出现的平均次数）


def _greedy_set_cover_lotto(pool: List[int], pick: int, n_tickets: int,
                            unit_price: int = 2) -> GreedyCoverPlan:
    """用贪心集合覆盖算法选择 n_tickets 注大乐透前区号码。

    目标：在有限的注数下，最大化覆盖的不同号码数量。
    策略：每轮选一个"包含最多未覆盖号码"的组合。

    pool: 可选号池（如前区 1-35）
    pick: 每注选几个号（5）
    n_tickets: 选多少注
    """
    from itertools import combinations
    import random as rnd

    all_numbers = set(pool)
    covered = set()
    selected = []

    # 预生成所有候选组合（太多时用采样替代）
    all_combos = list(combinations(pool, pick))
    if len(all_combos) > 50000:
        # 号池太大时，用随机采样 + 贪心
        # 每次从随机组合中选最优
        for _ in range(n_tickets):
            best_combo = None
            best_new = -1
            for _ in range(min(5000, len(all_combos))):
                combo = tuple(sorted(rnd.sample(pool, pick)))
                new_count = len(set(combo) - covered)
                if new_count > best_new:
                    best_new = new_count
                    best_combo = combo
                if best_new == pick:  # 全是新号，不可能更好
                    break
            if best_combo is None:
                best_combo = tuple(sorted(rnd.sample(pool, pick)))
            selected.append(list(best_combo))
            covered.update(best_combo)
    else:
        remaining = set(all_combos)
        for _ in range(n_tickets):
            best_combo = None
            best_new = -1
            for combo in remaining:
                new_count = len(set(combo) - covered)
                if new_count > best_new:
                    best_new = new_count
                    best_combo = combo
                    if best_new == pick:
                        break
            if best_combo is None:
                # 都用完了，从全集中随机
                best_combo = rnd.choice(all_combos)
            selected.append(list(best_combo))
            covered.update(best_combo)

    # 计算冗余率
    from collections import Counter
    num_counter = Counter()
    for ticket in selected:
        for n in ticket:
            num_counter[n] += 1
    redundancy = sum(num_counter.values()) / len(num_counter) if num_counter else 1

    return GreedyCoverPlan(
        selected_tickets=selected,
        total_bets=n_tickets,
        total_cost=n_tickets * unit_price,
        covered_numbers=covered,
        coverage_rate=len(covered) / len(pool),
        redundancy=round(redundancy, 2),
    )


def greedy_cover_lotto_front(front_pool_size: int = 35, front_pick: int = 5,
                              n_tickets: int = 12, unit_price: int = 2) -> GreedyCoverPlan:
    """大乐透前区贪心覆盖。"""
    pool = list(range(1, front_pool_size + 1))
    return _greedy_set_cover_lotto(pool, front_pick, n_tickets, unit_price)


def greedy_cover_report(game: str, max_budget: int = 200) -> dict:
    """为某玩法生成贪心覆盖优化报告。

    枚举不同的注数，展示覆盖率和冗余率的权衡。
    """
    cfg = GAMES[game]
    unit = BET_UNIT_PRICE.get(game, 2)
    plans = []

    if cfg["type"] == "lotto":
        pool = cfg["front_pool"]
        pick = cfg["front_pick"]
        max_tickets = max_budget // unit
        for n in [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]:
            if n > max_tickets:
                continue
            plan = _greedy_set_cover_lotto(pool, pick, n, unit)
            plans.append({
                "n_tickets": n,
                "total_cost": plan.total_cost,
                "covered_numbers": sorted(plan.covered_numbers),
                "coverage_rate": round(plan.coverage_rate, 3),
                "redundancy": plan.redundancy,
            })
    else:
        # digit 玩法：每位贪心覆盖
        positions = cfg["positions"]
        max_tickets = max_budget // unit
        for n in [1, 2, 3, 5, 10, 15, 20, 30]:
            if n > max_tickets:
                continue
            # 每位独立贪心：每位选 n 个数字，使覆盖的数字最大化（0-9）
            covered_positions = []
            for pos in range(positions):
                pool = list(range(10))
                plan = _greedy_set_cover_lotto(pool, 1, min(n, 10), unit)
                covered_positions.append(sorted(plan.covered_numbers))
            # 总覆盖 = 所有位覆盖的并集
            all_covered = set()
            for cp in covered_positions:
                all_covered.update(cp)
            plans.append({
                "n_tickets": n,
                "total_cost": n * unit,
                "covered_per_position": covered_positions,
                "total_unique_digits": len(all_covered),
                "coverage_rate": round(len(all_covered) / (positions * 10), 3),
                "redundancy": 1.0,
            })

    return {
        "game": game,
        "max_budget": max_budget,
        "plans": plans,
        "recommendation": _pick_best_cover_plan(plans, max_budget),
    }


def _pick_best_cover_plan(plans: List[dict], max_budget: int) -> dict:
    """选最佳覆盖方案：在预算内选覆盖率最高且冗余度适中的。"""
    if not plans:
        return {}
    # 在预算内优先选覆盖率高的，覆盖率相同时选注数少的
    valid = [p for p in plans if p["total_cost"] <= max_budget]
    if not valid:
        return plans[0]
    # 按覆盖率降序，冗余度升序
    valid.sort(key=lambda x: (-x["coverage_rate"], x.get("redundancy", 0)))
    return valid[0]


# ---------------- 资金管理 ----------------

# 体彩合理投注的参考阈值（元/月）。超过即触发提醒。
# 这是基于"娱乐性支出不应超过可支配收入 1%"的常识设定，非投资建议。
DEFAULT_MONTHLY_LIMIT = 200
DEFAULT_CHASE_LIMIT = 5   # 连续追号期数上限（止损）


def budget_warning(monthly_spent: int, consecutive_losses: int = 0,
                   monthly_limit: int = DEFAULT_MONTHLY_LIMIT,
                   chase_limit: int = DEFAULT_CHASE_LIMIT) -> dict:
    """资金管理提醒。返回 {level, message}。"""
    msgs = []
    level = "ok"
    if monthly_spent >= monthly_limit * 2:
        level = "danger"
        msgs.append(f"本月已投入 ¥{monthly_spent}，是建议上限 ¥{monthly_limit} 的 2 倍，请立即暂停。")
    elif monthly_spent >= monthly_limit:
        level = "warning"
        msgs.append(f"本月已投入 ¥{monthly_spent}，已达建议上限 ¥{monthly_limit}，建议停止本月投注。")
    elif monthly_spent >= monthly_limit * 0.7:
        level = "info"
        msgs.append(f"本月已投入 ¥{monthly_spent}，接近建议上限 ¥{monthly_limit}，注意控制。")

    if consecutive_losses >= chase_limit:
        if level != "danger":
            level = "warning"
        msgs.append(f"已连续 {consecutive_losses} 期未中，达到追号止损线 {chase_limit}，建议停止追号。")

    if not msgs:
        msgs.append(f"本月投入 ¥{monthly_spent}，在合理范围内。理性购彩，量力而行。")
    return {"level": level, "messages": msgs,
            "monthly_limit": monthly_limit, "chase_limit": chase_limit}


if __name__ == "__main__":
    # 自检
    print("===== 排列3 单注概率 =====")
    print(digit_single_prob(3))
    print("\n===== 排列3 小复式(2,2,2) =====")
    print(digit_coverage([2, 2, 2]))
    print("\n===== 排列3 中复式(3,3,3) =====")
    print(digit_coverage([3, 3, 3]))
    print("\n===== 大乐透单注(5+2) =====")
    print(lotto_coverage(5, 2))
    print("\n===== 大乐透 8+4 复式 =====")
    print(lotto_coverage(8, 4))
    print("\n===== 预算 ¥54 买排列3 =====")
    p = recommend_budget("pl3", 54)
    print("推荐:", p.recommended)
    print("备选:", p.alternatives)
    print("\n===== 预算 ¥20 买大乐透 =====")
    p = recommend_budget("dlt", 20)
    print("推荐:", p.recommended)
