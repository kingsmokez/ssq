# -*- coding: utf-8 -*-
"""体彩推荐分析系统 — Flask 应用入口。

⚠️ 免责声明：彩票每次开奖均为独立随机事件，任何算法都无法真正预测中奖号码。
   本系统基于历史统计做参考推荐，仅供娱乐与研究，不构成投注建议。
"""

import functools
import csv
import io

from flask import (
    Flask, render_template, request, redirect, url_for, session,
    jsonify, flash, abort, Response,
)

from config import (
    SECRET_KEY, LOGIN_PASSWORD, HOST, PORT, GAMES, GAME_ORDER,
    DISCLAIMER, BET_UNIT_PRICE,
)
from lottery import models
from lottery.fetcher import refresh_all
from lottery.analyzer import analyze
from lottery.recommender import recommend, recommend_and_save
from lottery.backtester import backtest_all, backtest_as_dict

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["JSON_AS_ASCII"] = False


# ---------------- 鉴权 ----------------

def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        pwd = request.form.get("password", "")
        if pwd == LOGIN_PASSWORD:
            session["logged_in"] = True
            nxt = request.args.get("next") or url_for("dashboard")
            return redirect(nxt)
        flash("密码错误，请重试。", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- 页面路由 ----------------

@app.route("/")
@login_required
def dashboard():
    latest = models.get_all_games_latest()
    counts = {g: models.count_draws(g) for g in GAME_ORDER}
    recos = {g: models.latest_recommendations(g, "single") for g in GAME_ORDER}
    logs = models.recent_refresh_logs(limit=6)
    return render_template(
        "dashboard.html", games=GAMES, game_order=GAME_ORDER,
        latest=latest, counts=counts, recos=recos, logs=logs,
        disclaimer=DISCLAIMER, active="dashboard")


@app.route("/game/<game>")
@login_required
def detail(game):
    if game not in GAMES:
        abort(404)
    days = int(request.args.get("days", 730))
    gs = analyze(game, days=days)
    single = models.latest_recommendations(game, "single")
    small = models.latest_recommendations(game, "small")
    medium = models.latest_recommendations(game, "medium")
    recent = models.fetch_draws(game, limit=20)
    return render_template(
        "detail.html", game=game, games=GAMES, gs=gs,
        single=single, small=small, medium=medium, recent=recent,
        days=days, disclaimer=DISCLAIMER, active=game)


@app.route("/history")
@login_required
def history():
    game = request.args.get("game", "dlt")
    if game not in GAMES:
        game = "dlt"
    q = request.args.get("q", "").strip()
    rows = models.fetch_draws(game, limit=200)
    if q:
        rows = [r for r in rows if q in str(r["issue"]) or q in str(r["draw_date"])]
    return render_template(
        "history.html", game=game, games=GAMES, rows=rows, q=q,
        disclaimer=DISCLAIMER, active="history")


@app.route("/randomness")
@login_required
def randomness_view():
    return render_template(
        "randomness.html", games=GAMES, game_order=GAME_ORDER,
        disclaimer=DISCLAIMER, active="randomness")


@app.route("/backtest")
@login_required
def backtest():
    """多期回测：将历史推荐与对应期实际开奖对比，展示累计命中率。"""
    results = {}
    for game in GAME_ORDER:
        # 按时间正序排列，方便查找推荐时间之后的下一期开奖
        draws_asc = models.fetch_draws(game, limit=500, order_desc=False)
        draws_desc = models.fetch_draws(game, limit=200)

        # 获取所有推荐批次
        batches = models.all_recommendation_batches(game, limit=30)

        # 按创建时间分组
        from collections import defaultdict
        batch_groups = defaultdict(dict)
        for (created_at, mode), items in batches.items():
            batch_groups[created_at][mode] = items

        # 取最近 10 批
        sorted_batches = sorted(batch_groups.items(), key=lambda x: x[0], reverse=True)[:10]
        hits_per_batch = []

        for created_at, modes in sorted_batches:
            # 找到推荐创建日期之后的首期开奖
            next_draw = None
            for d in draws_asc:
                dd = d.get("draw_date", "")
                if dd and dd[:10] >= created_at[:10]:
                    next_draw = d
                    break

            if not next_draw:
                continue

            front_actual = set(next_draw["front"])
            back_actual = set(next_draw["back"])

            batch_hits = {
                "created_at": created_at,
                "draw_issue": next_draw["issue"],
                "draw_date": next_draw["draw_date"][:10] if next_draw["draw_date"] else "",
                "groups": [],
            }

            single_recos = modes.get("single", [])
            for grp in single_recos:
                picks = grp["picks"]
                if "front" in picks:
                    f_hit = len(set(picks["front"]) & front_actual)
                    b_hit = len(set(picks["back"]) & back_actual)
                    batch_hits["groups"].append({
                        "label": grp["label"],
                        "front_hit": f_hit, "back_hit": b_hit,
                        "total_front": len(picks["front"]),
                        "total_back": len(picks["back"]),
                    })
                else:
                    match = sum(
                        1 for i, d in enumerate(picks["digits"])
                        if i < len(next_draw["front"]) and next_draw["front"][i] == d
                    )
                    batch_hits["groups"].append({
                        "label": grp["label"], "position_match": match,
                        "total_positions": len(picks["digits"]),
                    })
            hits_per_batch.append(batch_hits)

        # 计算汇总命中率
        total_front_hits = total_front_picks = 0
        total_back_hits = total_back_picks = 0
        total_pos_matches = total_pos_picks = 0
        for bh in hits_per_batch:
            for g in bh["groups"]:
                if "front_hit" in g:
                    total_front_hits += g["front_hit"]
                    total_front_picks += g["total_front"]
                    total_back_hits += g["back_hit"]
                    total_back_picks += g["total_back"]
                else:
                    total_pos_matches += g["position_match"]
                    total_pos_picks += g["total_positions"]

        results[game] = {
            "batches": hits_per_batch,
            "latest": draws_desc[0] if draws_desc else None,
            "draws_count": len(draws_desc),
            "aggregate": {
                "front_hit_rate": round(total_front_hits / total_front_picks, 3) if total_front_picks else 0,
                "back_hit_rate": round(total_back_hits / total_back_picks, 3) if total_back_picks else 0,
                "pos_hit_rate": round(total_pos_matches / total_pos_picks, 3) if total_pos_picks else 0,
                "batches_tested": len(hits_per_batch),
            },
        }

    return render_template(
        "backtest.html", games=GAMES, game_order=GAME_ORDER,
        results=results, disclaimer=DISCLAIMER, active="backtest")


# ---------------- API ----------------

@app.route("/api/refresh", methods=["POST"])
@login_required
def api_refresh():
    """触发抓取 + 重算推荐。"""
    try:
        res = refresh_all()
        summary = {}
        for g, v in res.items():
            if isinstance(v, tuple):
                summary[g] = {"status": "error", "message": v[1], "new": 0}
            else:
                summary[g] = {"status": "ok", "new": v}
                # 重算并保存推荐
                gs = analyze(g)
                recommend_and_save(g, gs)
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/stats/<game>")
@login_required
def api_stats(game):
    """返回某玩法的图表数据（频率/遗漏/冷热）。"""
    if game not in GAMES:
        abort(404)
    days = int(request.args.get("days", 730))
    gs = analyze(game, days=days)
    cfg = GAMES[game]
    data = {"game": game, "name": gs.name, "window_draws": gs.window_draws,
            "total_draws": gs.total_draws,
            "latest_issue": gs.latest_issue, "latest_date": gs.latest_date}
    if cfg["type"] == "lotto":
        data["front"] = _number_stat_payload(gs.front_stats)
        data["back"] = _number_stat_payload(gs.back_stats)
        # 共生矩阵（前区号码对共生率）
        cooc = gs.cooccur
        if cooc:
            cooc_list = []
            for a in sorted(cooc):
                for b in sorted(cooc[a]):
                    if a < b:
                        cooc_list.append({"a": a, "b": b, "rate": round(cooc[a][b], 4)})
            data["cooccurrence"] = cooc_list
    else:
        data["positions"] = {str(p): _number_stat_payload(gs.position_stats[p])
                             for p in gs.position_stats}
    return jsonify(data)


@app.route("/api/omit_trend/<game>")
@login_required
def api_omit_trend(game):
    """返回某玩法遗漏趋势数据（最近50期，前10个号码）。"""
    if game not in GAMES:
        abort(404)
    days = int(request.args.get("days", 730))
    cfg = GAMES[game]
    draws = models.fetch_draws(game, limit=50, order_desc=False)

    if cfg["type"] == "lotto":
        pool = cfg["front_pool"]
        # 取遗漏值最高的前10个号码
        gs = analyze(game, days=days)
        top_omit = sorted(gs.front_stats, key=lambda n: -gs.front_stats[n].omission)[:10]
        numbers = top_omit
    else:
        # digit: 取第1位遗漏最高的前5个数字
        gs = analyze(game, days=days)
        pos0 = gs.position_stats.get(0, {})
        top_omit = sorted(pos0, key=lambda n: -pos0[n].omission)[:5]
        numbers = top_omit
        pool = list(range(10))

    # 计算每期每个号码的累计遗漏
    trend = {n: [] for n in numbers}
    issues = []
    # 从远到近遍历，计算每个号码到当前期的遗漏
    seen = {n: 0 for n in numbers}
    for r in draws:
        issues.append(r["issue"][-5:])  # 取期号后5位
        front_set = set(r.get("front", []))
        for n in numbers:
            if n in front_set:
                seen[n] = 0
            else:
                seen[n] += 1
            trend[n].append(seen[n])

    return jsonify({
        "game": game,
        "issues": issues,
        "numbers": numbers,
        "trend": {str(n): trend[n] for n in numbers},
    })


def _number_stat_payload(stats):
    """把 NumberStat dict 转成前端友好的数组。"""
    out = []
    for n in sorted(stats):
        s = stats[n]
        out.append({
            "number": n, "freq": round(s.freq, 4), "omission": s.omission,
            "avg_omission": round(s.avg_omission, 1),
            "omission_ratio": round(s.omission_ratio, 3),
            "momentum": round(s.momentum, 4),
            "cold_hot": s.cold_hot, "warm_tendency": s.warm_tendency,
            "score": None,
        })
    return out


@app.route("/api/backtest")
@login_required
def api_backtest():
    """运行逐期回测，返回模型 vs 随机基线的命中率对比。
    耗时约 30-60 秒（3 玩法 × 119 测试点）。"""
    n = int(request.args.get("n", 120))
    full = request.args.get("full", "0") == "1"  # ?full=1 启用全管线回测
    try:
        if full:
            from lottery.backtester import backtest_full_pipeline_all, full_pipeline_as_dict
            results = backtest_full_pipeline_all(n_test=n)
            payload = {g: full_pipeline_as_dict(r) for g, r in results.items()}
        else:
            results = backtest_all(n_test=n)
            payload = {g: backtest_as_dict(r) for g, r in results.items()}
        return jsonify({"ok": True, "results": payload, "full_pipeline": full})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/backtest/full")
@login_required
def api_backtest_full():
    """全管线回测：测试真实 recommend() 输出的所有6组+复式。
    耗时较长（~90秒），但测试的是用户实际看到的推荐。"""
    n = int(request.args.get("n", 80))
    try:
        from lottery.backtester import backtest_full_pipeline_all, full_pipeline_as_dict
        results = backtest_full_pipeline_all(n_test=n)
        payload = {g: full_pipeline_as_dict(r) for g, r in results.items()}
        return jsonify({"ok": True, "results": payload})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/probability/<game>")
@login_required
def api_probability(game):
    """返回某玩法当前推荐的概率提示（诚实呈现）。

    核心信息：单注"至少中一位"和"全不中"的概率。
    用于在前端提醒用户"全不中"是大概率事件，不要归因成模型不准。
    """
    if game not in GAMES:
        abort(404)
    from lottery.ev_calc import digit_single_prob, lotto_coverage
    cfg = GAMES[game]
    if cfg["type"] == "digit":
        prob = digit_single_prob(cfg["positions"])
        return jsonify({
            "game": game, "type": "digit",
            "p_none": round(prob["p_none"], 4),
            "p_at_least_one": round(prob["p_at_least_one"], 4),
            "p_all_hit": prob["p_all_hit"],
            "positions": cfg["positions"],
            "hint": (f"排列{cfg['positions']} 单注：3位全不中的概率 ≈ "
                     f"{prob['p_none']:.1%}，是最可能的结果。"
                     f"直选中奖概率恒为 1/{10**cfg['positions']}。"),
        })
    else:
        cov = lotto_coverage(cfg["front_pick"], cfg["back_pick"],
                             BET_UNIT_PRICE.get(game, 2),
                             len(cfg["front_pool"]), len(cfg["back_pool"]),
                             cfg["front_pick"], cfg["back_pick"])
        return jsonify({
            "game": game, "type": "lotto",
            "p_any_hit": round(cov.p_any_hit, 4),
            "p_none": round(1 - cov.p_any_hit, 4),
            "p_first_prize": cov.p_first_prize,
            "p_front_at_least_one": round(cov.p_front_at_least_one, 4),
            "p_back_at_least_one": round(cov.p_back_at_least_one, 4),
            "hint": (f"大乐透单注至少中1个的概率 ≈ {cov.p_any_hit:.1%}，"
                     f"但中一等奖概率仅 ≈ {cov.p_first_prize:.2e}（约1/{1/cov.p_first_prize:.0f}）。"),
        })


@app.route("/api/randomness/<game>")
@login_required
def api_randomness(game):
    """对某玩法的历史开奖做全套随机性检测。

    包含：卡方拟合优度检验、游程检验、频率平衡分析。
    结果用于在前端展示"彩票是否真的是随机的"。
    """
    if game not in GAMES:
        abort(404)
    from lottery.randomness import test_draws
    draws = models.fetch_draws(game, limit=500, order_desc=True)
    cfg = GAMES[game]
    if cfg["type"] == "lotto":
        pool_size = len(cfg["front_pool"])
        draw_pick = cfg["front_pick"]
    else:
        pool_size = 10
        draw_pick = 1
    result = test_draws(draws, pool_size, draw_pick)
    return jsonify(result)



@app.route("/api/ev/<game>")
@login_required
def api_ev(game):
    """预算×覆盖率计算器：给定预算，推荐最划算的复式档位。

    参数 budget=预算（元）。
    返回推荐档位 + 备选，含"至少中一位"概率、注数、成本。
    """
    if game not in GAMES:
        abort(404)
    budget = int(request.args.get("budget", 20))
    if budget < 2 or budget > 100000:
        return jsonify({"ok": False, "error": "预算需在 2~100000 元之间"}), 400
    from lottery.ev_calc import recommend_budget
    from lottery.ev_calc import greedy_cover_report as gcr
    plan = recommend_budget(game, budget)
    # 同时返回贪心覆盖报告
    cover_report = gcr(game, max_budget=budget)
    def _ser(d):
        out = dict(d)
        if "value_ratio" in out:
            out["value_ratio"] = round(out["value_ratio"], 6)
        return out
    return jsonify({
        "ok": True, "game": game, "budget": budget,
        "recommended": _ser(plan.recommended),
        "alternatives": [_ser(a) for a in plan.alternatives],
        "greedy_cover": cover_report,
    })


@app.route("/api/export/draws/<game>")
@login_required
def api_export_draws(game):
    """导出某玩法历史开奖数据为 CSV。"""
    if game not in GAMES:
        abort(404)
    rows = models.fetch_draws(game, limit=5000)
    cfg = GAMES[game]
    output = io.StringIO()
    if cfg["type"] == "lotto":
        writer = csv.writer(output)
        writer.writerow(["期号", "日期", "前区1", "前区2", "前区3", "前区4", "前区5",
                         "后区1", "后区2"])
        for r in rows:
            front = r.get("front", [])
            back = r.get("back", [])
            writer.writerow([
                r["issue"], r.get("draw_date", "")[:10],
                *front, *back,
            ])
    else:
        writer = csv.writer(output)
        positions = cfg["positions"]
        writer.writerow(["期号", "日期"] + [f"第{i+1}位" for i in range(positions)])
        for r in rows:
            front = r.get("front", [])
            writer.writerow([r["issue"], r.get("draw_date", "")[:10], *front])
    filename = f"{game}_draws_{len(rows)}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv; charset=utf-8-sig",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/api/export/recommendations/<game>")
@login_required
def api_export_recommendations(game):
    """导出某玩法最新推荐为 CSV。"""
    if game not in GAMES:
        abort(404)
    cfg = GAMES[game]
    output = io.StringIO()
    writer = csv.writer(output)
    if cfg["type"] == "lotto":
        writer.writerow(["组号", "模式", "标签", "前区", "后区", "注数", "金额(元)", "推荐理由"])
    else:
        writer.writerow(["组号", "模式", "标签", "各位数字", "注数", "金额(元)", "推荐理由"])

    for mode in ("single", "small", "medium"):
        recos = models.latest_recommendations(game, mode)
        for r in recos:
            picks = r.get("picks", {})
            if "front" in picks:
                front_str = " ".join(str(n).zfill(2) for n in picks["front"])
                back_str = " ".join(str(n).zfill(2) for n in picks["back"])
                nums_str = f"{front_str} + {back_str}"
            else:
                digits = picks.get("digits", picks.get("digits_grid", []))
                if isinstance(digits[0], list) if digits else False:
                    nums_str = " | ".join("/".join(str(d) for d in pos) for pos in digits)
                else:
                    nums_str = " ".join(str(d) for d in digits)
            writer.writerow([
                r.get("group_index", ""), r.get("mode", mode),
                r.get("label", ""), nums_str,
                r.get("bets", 1), r.get("cost", 2),
                r.get("reason", ""),
            ])
    filename = f"{game}_recommendations.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv; charset=utf-8-sig",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", disclaimer=DISCLAIMER,
                           code=404, msg="页面不存在"), 404


def main():
    models.init_db()
    print("=" * 50)
    print("体彩推荐分析系统")
    print(f"  访问: http://{HOST}:{PORT}")
    print(f"  密码: {LOGIN_PASSWORD}")
    print("  按 Ctrl+C 退出")
    print("=" * 50)
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()
