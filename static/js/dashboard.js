/* 仪表盘 — 刷新数据并计算推荐 */
(function () {
    var btn = document.getElementById('refreshBtn');
    var status = document.getElementById('refreshStatus');
    if (!btn) return;

    btn.addEventListener('click', function () {
        if (!confirm('即将从 500.com 抓取最新数据并重新计算推荐，可能需要几秒到十几秒。继续？')) return;

        btn.disabled = true;
        var origText = btn.innerHTML;
        btn.innerHTML = '<span class="spinner-inline"></span> 抓取中...';
        status.className = 'alert alert-info';
        status.textContent = '正在抓取最新数据并计算推荐，请稍候...';

        fetch('/api/refresh', { method: 'POST' })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.ok) {
                    var lines = [];
                    var summary = data.summary || {};
                    Object.keys(summary).forEach(function (g) {
                        var s = summary[g];
                        var name = { dlt: '大乐透', pl3: '排列3', pl5: '排列5' }[g] || g;
                        if (s.status === 'ok') {
                            lines.push(name + '：新增 ' + s.new + ' 期，推荐已更新');
                        } else {
                            lines.push(name + '：失败 - ' + s.message);
                        }
                    });
                    status.className = 'alert alert-success';
                    status.innerHTML = '✅ 完成！' + lines.join('；') + '<br>页面即将刷新...';
                    // 自动重载页面，确保首页推荐与详情页一致
                    setTimeout(function () { window.location.reload(); }, 1200);
                } else {
                    status.className = 'alert alert-danger';
                    status.textContent = '❌ 失败：' + (data.error || '未知错误');
                }
            })
            .catch(function (err) {
                status.className = 'alert alert-danger';
                status.textContent = '❌ 网络错误：' + err;
            })
            .finally(function () {
                btn.disabled = false;
                btn.innerHTML = origText;
            });
    });

    // ---------- 资金管理 ----------
    var MONTHLY_LIMIT = 200;
    var spentInput = document.getElementById('monthSpent');
    var saveBtn = document.getElementById('saveSpentBtn');
    var warnBox = document.getElementById('budgetWarnBox');
    if (!spentInput || !warnBox) return;

    // 读取本月记录（localStorage 按年月分桶）
    var ym = new Date().toISOString().slice(0, 7); // YYYY-MM
    var key = 'bocai_spent_' + ym;
    var saved = parseInt(localStorage.getItem(key) || '0', 10);
    if (saved > 0) spentInput.value = saved;

    function render() {
        var v = parseInt(spentInput.value || '0', 10);
        var pct = Math.round(v / MONTHLY_LIMIT * 100);
        var cls = 'alert alert-info small mb-0 py-2';
        var msg;
        if (v >= MONTHLY_LIMIT * 2) {
            cls = 'alert alert-danger small mb-0 py-2';
            msg = '🚨 已投入 ¥' + v + '，是建议上限 ¥' + MONTHLY_LIMIT +
                  ' 的 ' + pct + '%，请立即暂停投注。';
        } else if (v >= MONTHLY_LIMIT) {
            cls = 'alert alert-warning small mb-0 py-2';
            msg = '⚠️ 已投入 ¥' + v + '，达建议上限，建议本月停止投注。';
        } else if (v >= MONTHLY_LIMIT * 0.7) {
            cls = 'alert alert-warning small mb-0 py-2';
            msg = '⏳ 已投入 ¥' + v + '（' + pct + '%），接近上限，注意控制。';
        } else {
            msg = '✅ 已投入 ¥' + v + '（' + pct + '% 上限），在合理范围内。理性购彩，量力而行。';
        }
        warnBox.className = cls;
        warnBox.textContent = msg;
    }

    render();
    if (saveBtn) saveBtn.addEventListener('click', function () {
        var v = parseInt(spentInput.value || '0', 10);
        if (v < 0) v = 0;
        localStorage.setItem(key, String(v));
        render();
    });
    spentInput.addEventListener('input', render);
})();
