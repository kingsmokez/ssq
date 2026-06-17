/* 详情页 — 频率/遗漏/共生热力图/遗漏趋势图表（Chart.js） */
(function () {
    var api = window.STATS_API;
    if (!api || typeof Chart === 'undefined') return;

    var coldHotColor = { '热': '#e63946', '冷': '#2a6fdb', '平': '#adb5bd' };

    fetch(api)
        .then(function (r) { return r.json(); })
        .then(function (data) {
            renderCharts(data);
            if (data.cooccurrence) renderCooccurChart(data.cooccurrence, data.front);
        })
        .catch(function (e) { console.warn('stats load failed', e); });

    // 遗漏趋势图（独立请求）
    var trendApi = window.OMIT_TREND_API || api.replace('/api/stats/', '/api/omit_trend/');
    fetch(trendApi)
        .then(function (r) { return r.json(); })
        .then(function (data) { renderOmitTrend(data); })
        .catch(function (e) { console.warn('omit trend load failed', e); });

    function renderStats(labels, freqs, omits, ch, ctxFreq, ctxOmit, color) {
        var bg = labels.map(function (_, i) { return coldHotColor[ch[i]] || color; });
        new Chart(ctxFreq, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: '出现频率',
                    data: freqs,
                    backgroundColor: bg,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                return '频率 ' + (ctx.parsed.y * 100).toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: { y: { beginAtZero: true, ticks: { callback: function (v) { return (v * 100).toFixed(0) + '%'; } } } }
            }
        });

        new Chart(ctxOmit, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: '当前遗漏期数',
                    data: omits,
                    backgroundColor: bg,
                    borderWidth: 0
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: function (ctx) { return '遗漏 ' + ctx.parsed.x + ' 期'; } } }
                },
                scales: { x: { beginAtZero: true } }
            }
        });
    }

    function renderCharts(data) {
        var ctxF = document.getElementById('freqChart');
        var ctxO = document.getElementById('omitChart');
        if (!ctxF || !ctxO) return;

        if (data.front) {
            // lotto: 前区柱状图 + 后区可选
            var arr = data.front;
            var labels = arr.map(function (d) { return ('0' + d.number).slice(-2); });
            var freqs = arr.map(function (d) { return d.freq; });
            var omits = arr.map(function (d) { return d.omission; });
            var ch = arr.map(function (d) { return d.cold_hot; });
            // 遗漏图按遗漏值倒序
            var omitSorted = arr.map(function (d, i) {
                return { label: labels[i], v: d.omission, ch: d.cold_hot };
            }).sort(function (a, b) { return b.v - a.v; });

            renderStats(labels, freqs, omits, ch, ctxF, ctxO, '#e63946');
            // 重画遗漏图为水平条形（遗漏降序）— 覆盖上面
            var c2 = ctxO.getContext('2d');
            Chart.getChart(ctxO).destroy();
            new Chart(ctxO, {
                type: 'bar',
                data: {
                    labels: omitSorted.map(function (x) { return x.label; }),
                    datasets: [{
                        data: omitSorted.map(function (x) { return x.v; }),
                        backgroundColor: omitSorted.map(function (x) { return coldHotColor[x.ch] || '#adb5bd'; })
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    plugins: { legend: { display: false }, tooltip: { callbacks: { label: function (c) { return '遗漏 ' + c.parsed.x + ' 期'; } } } },
                    scales: { x: { beginAtZero: true } }
                }
            });
        } else if (data.positions) {
            // digit: 默认显示第1位
            var pos0 = data.positions['0'];
            var labels = pos0.map(function (d) { return d.number; });
            var freqs = pos0.map(function (d) { return d.freq; });
            var omits = pos0.map(function (d) { return d.omission; });
            var ch = pos0.map(function (d) { return d.cold_hot; });
            renderStats(labels, freqs, omits, ch, ctxF, ctxO, '#343a40');
        }
    }

    /**
     * 共生率热力图 — 用 Canvas 手绘 35×35 的网格（Chart.js 无原生热力图）。
     */
    function renderCooccurChart(cooccurData, frontStats) {
        var canvas = document.getElementById('cooccurChart');
        if (!canvas || !cooccurData || cooccurData.length === 0) return;

        // 构建共生率矩阵
        var numbers = frontStats.map(function (d) { return d.number; });
        var n = numbers.length;
        var maxNum = Math.max.apply(null, numbers);
        var matrix = {};
        var maxRate = 0;
        cooccurData.forEach(function (d) {
            var key = d.a + '_' + d.b;
            matrix[key] = d.rate;
            if (d.rate > maxRate) maxRate = d.rate;
        });
        maxRate = maxRate || 1;

        // 用 Chart.js 的 matrix 插件（如果没有，用 canvas 手绘）
        var size = Math.min(30, Math.floor(800 / n));
        canvas.width = n * size + 60;
        canvas.height = n * size + 60;
        var ctx = canvas.getContext('2d');

        // 背景
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // 绘制网格
        for (var i = 0; i < n; i++) {
            for (var j = 0; j < n; j++) {
                var a = numbers[i], b = numbers[j];
                var x = 30 + j * size;
                var y = 30 + i * size;
                if (i === j) {
                    ctx.fillStyle = '#dee2e6'; // 对角线灰色
                } else {
                    var key = Math.min(a, b) + '_' + Math.max(a, b);
                    var rate = matrix[key] || 0;
                    var intensity = rate / maxRate;
                    // 从白色到深红色的渐变
                    var r = Math.round(255 - intensity * 200);
                    var g = Math.round(255 - intensity * 220);
                    var b2 = Math.round(255 - intensity * 200);
                    ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b2 + ')';
                }
                ctx.fillRect(x, y, size - 1, size - 1);
            }
        }

        // 标签
        ctx.fillStyle = '#495057';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        for (var i = 0; i < n; i++) {
            var label = ('0' + numbers[i]).slice(-2);
            ctx.fillText(label, 30 + i * size + size / 2, 25);
            ctx.save();
            ctx.translate(25, 30 + i * size + size / 2);
            ctx.fillText(label, 0, 0);
            ctx.restore();
        }

        // 色标
        var legendX = 30 + n * size + 10;
        var legendH = n * size;
        for (var i = 0; i < legendH; i++) {
            var intensity = 1 - i / legendH;
            var r = Math.round(255 - intensity * 200);
            var g = Math.round(255 - intensity * 220);
            var b2 = Math.round(255 - intensity * 200);
            ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b2 + ')';
            ctx.fillRect(legendX, 30 + i, 15, 1);
        }
        ctx.fillStyle = '#495057';
        ctx.textAlign = 'left';
        ctx.fillText((maxRate * 100).toFixed(0) + '%', legendX + 18, 38);
        ctx.fillText('0%', legendX + 18, 30 + legendH);
    }

    /**
     * 遗漏趋势折线图 — 展示前10个号码近50期的遗漏变化。
     */
    function renderOmitTrend(data) {
        var canvas = document.getElementById('omitTrendChart');
        if (!canvas || !data.issues || data.issues.length === 0) return;

        var colors = [
            '#e63946', '#2a6fdb', '#2a9d8f', '#e9c46a', '#f4a261',
            '#264653', '#6a4c93', '#1982c4', '#8ac926', '#ff595e'
        ];

        var datasets = data.numbers.map(function (n, i) {
            var label = ('0' + n).slice(-2);
            return {
                label: label,
                data: data.trend[String(n)] || [],
                borderColor: colors[i % colors.length],
                backgroundColor: 'transparent',
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.3,
            };
        });

        new Chart(canvas, {
            type: 'line',
            data: {
                labels: data.issues,
                datasets: datasets,
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { boxWidth: 12, font: { size: 10 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                return ctx.dataset.label + ': 遗漏 ' + ctx.parsed.y + ' 期';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { maxTicksLimit: 10, font: { size: 10 } },
                        title: { display: true, text: '期号' }
                    },
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: '遗漏期数' },
                        ticks: { font: { size: 10 } }
                    }
                }
            }
        });
    }
})();
