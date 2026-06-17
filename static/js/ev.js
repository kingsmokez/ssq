/* -*- 概率提示 + 预算计算器 -*-
 * 加载 /api/probability 渲染"概率真相"卡片；
 * 监听预算输入，调用 /api/ev 渲染最划算复式档位。
 * 纯前端展示，无状态。
 */
(function () {
    "use strict";

    function pct(p) { return (p * 100).toFixed(1) + "%"; }

    // ---------- 概率提示 ----------
    function loadProbability() {
        var body = document.getElementById("probHintBody");
        if (!body || !window.PROB_API) return;
        fetch(window.PROB_API)
            .then(function (r) { return r.json(); })
            .then(function (d) {
                if (!d) return;
                if (d.type === "digit") {
                    body.innerHTML =
                        '<div class="d-flex justify-content-around text-center mb-2">' +
                          '<div><div class="fs-4 text-danger">' + pct(d.p_none) + '</div>' +
                          '<div class="small text-muted">3位全不中</div></div>' +
                          '<div><div class="fs-4 text-success">' + pct(d.p_at_least_one) + '</div>' +
                          '<div class="small text-muted">至少中1位</div></div>' +
                          '<div><div class="fs-4 text-muted">1/' + Math.pow(10, d.positions) + '</div>' +
                          '<div class="small text-muted">直选中奖</div></div>' +
                        '</div>' +
                        '<div class="alert alert-warning small mb-0 py-2">' + d.hint + "</div>";
                } else {
                    body.innerHTML =
                        '<div class="d-flex justify-content-around text-center mb-2">' +
                          '<div><div class="fs-5 text-success">' + pct(d.p_any_hit) + '</div>' +
                          '<div class="small text-muted">至少中1个</div></div>' +
                          '<div><div class="fs-5 text-danger">' + pct(d.p_none) + '</div>' +
                          '<div class="small text-muted">全不中</div></div>' +
                          '<div><div class="fs-5 text-muted">1/' + Math.round(1 / d.p_first_prize).toLocaleString() + '</div>' +
                          '<div class="small text-muted">一等奖</div></div>' +
                        '</div>' +
                        '<div class="alert alert-warning small mb-0 py-2">' + d.hint + "</div>";
                }
            })
            .catch(function () {
                body.innerHTML = '<div class="text-danger small">概率数据加载失败。</div>';
            });
    }

    // ---------- 预算计算器 ----------
    function formatPick(r) {
        if (r.candidates_per_pos) {
            return "每位候选 [" + r.candidates_per_pos.join(", ") + "]";
        }
        return "前区选 " + r.front_pick + " + 后区选 " + r.back_pick;
    }

    function coverageOf(r) {
        return r.p_at_least_one != null ? r.p_at_least_one : r.p_all_hit;
    }

    function renderPlan(d) {
        var box = document.getElementById("evResult");
        if (!box) return;
        if (!d || !d.recommended) {
            box.innerHTML = '<div class="text-muted small">无可用方案。</div>';
            return;
        }
        var rec = d.recommended;
        var cov = rec.p_at_least_one != null ? rec.p_at_least_one
                 : rec.p_any_hit;
        var html = '<div class="alert alert-success py-2 mb-2">' +
                   '<strong>推荐：' + formatPick(rec) + "</strong><br>" +
                   '<span class="small">注数 ' + rec.bets + ' · 成本 ¥' + rec.cost +
                   ' · 至少中一位 <span class="text-success fw-bold">' + pct(cov) + "</span></span>" +
                   "</div>";
        if (d.alternatives && d.alternatives.length) {
            html += '<div class="small text-muted mb-1">备选档位：</div><ul class="list-group list-group-flush">';
            d.alternatives.forEach(function (a) {
                var ac = a.p_at_least_one != null ? a.p_at_least_one : a.p_any_hit;
                html += '<li class="list-group-item px-0 py-1 small d-flex justify-content-between">' +
                        '<span>' + formatPick(a) + "（¥" + a.cost + "）</span>" +
                        "<span>至少中一位 " + pct(ac) + "</span></li>";
            });
            html += "</ul>";
        }
        html += '<div class="alert alert-light small mt-2 mb-0 border">⚠️ ' +
                "「至少中一位」提升靠加钱买更多组合，但直选中奖概率不会因此显著提高。" +
                "预算是用来「买到更多组合」的，不是用来「买准」的。</div>";
        box.innerHTML = html;
    }

    function calcBudget() {
        var input = document.getElementById("budgetInput");
        var box = document.getElementById("evResult");
        if (!input || !window.EV_API) return;
        var budget = parseInt(input.value, 10);
        if (!budget || budget < 2) {
            box.innerHTML = '<div class="text-danger small">预算至少 ¥2。</div>';
            return;
        }
        box.innerHTML = '<div class="text-muted small">计算中...</div>';
        fetch(window.EV_API + "?budget=" + budget)
            .then(function (r) { return r.json(); })
            .then(function (d) {
                if (d.ok === false) {
                    box.innerHTML = '<div class="text-danger small">' + (d.error || "错误") + "</div>";
                    return;
                }
                renderPlan(d);
            })
            .catch(function () {
                box.innerHTML = '<div class="text-danger small">计算失败，请重试。</div>';
            });
    }

    document.addEventListener("DOMContentLoaded", function () {
        loadProbability();
        var btn = document.getElementById("budgetCalcBtn");
        if (btn) btn.addEventListener("click", calcBudget);
        var inp = document.getElementById("budgetInput");
        if (inp) inp.addEventListener("keydown", function (e) {
            if (e.key === "Enter") { e.preventDefault(); calcBudget(); }
        });
    });
})();
