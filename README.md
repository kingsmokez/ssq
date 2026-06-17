# 体彩推荐分析系统

基于 **2000+期真实历史数据** 的体彩（超级大乐透 / 排列3 / 排列5）号码推荐与回测系统。

## ⚠️ 免责声明

彩票每次开奖均为**独立随机事件**，任何算法都无法真正预测中奖号码。本系统基于历史统计做参考推荐，仅供娱乐与研究，不构成投注建议。

## 功能

- 🎯 **6组单选推荐** + 2档复式 — 均衡热号/冷号回补/趋势加速/组合合规/幸运组合/纯随机基线 + 集成混合投票
- 🔬 **全管线回测** — 199+测试点逐期验证，ML-GBM / Stacking / ARIMA / Thompson采样 vs 随机基线
- 📊 **统计检验** — Fisher精确检验、贝叶斯Beta-Binomial、卡方检验、Wilson置信区间
- 🎲 **随机性检测** — 卡方拟合优度、游程检验、频率平衡分析
- 📈 **Chart.js可视化** — 命中分布直方图、累积趋势图、频率热力图
- 🤖 **ML评分引擎** — Gradient Boosting / Random Forest / Logistic Regression 三模型
- 🔄 **Stacking集成** — ML + 传统 + ARIMA 多评分器加权融合
- 📐 **间距分析** — 大乐透前区gap分布 + 组合约束优化
- 📅 **星期几效应** — 按周一/三/六分段统计

## 技术栈

- **后端**: Python 3.10+ / Flask
- **ML**: scikit-learn (GBM/RF/LR)
- **数据**: SQLite / 500.com 抓取
- **前端**: Bootstrap 5 / Chart.js

## 快速开始

```bash
# 1. 安装依赖
pip install flask scikit-learn numpy

# 2. 初始化数据库 & 抓取历史数据
python -m lottery.fetcher

# 3. 启动服务
python app.py
# 访问 http://127.0.0.1:5000
# 默认密码: admin123
```

## 回测运行

```bash
# 全管线回测（测试真实推荐输出）
python -c "from lottery import detailed_backtest; detailed_backtest.print_detailed_report('dlt', n_test=200)"

# ML评分器回测
python -c "from lottery import ml_scorer; ml_scorer.MLScorer('dlt').backtest(n_test=100)"

# 超参网格搜索
python -m lottery.hyperopt dlt 60
```

## 回测结论（2000+期数据，199测试点）

| 方法 | DLT | PL3 | PL5 |
|------|-----|-----|-----|
| ML-GBM top-7 | 14.86% | - | - |
| 随机基线 top-7 | 14.43% | - | - |
| 传统 top-7 | 13.86% | - | - |
| 复式 medium | 13.63% | **29.31%** | **29.65%** |
| 幸运组合 top-5 | 14.47% | 11.56% | 9.35% |
| 冷号回补 top-5 | **10.95%** ❌ | **7.33%** ❌ | **7.60%** ❌ |

> 所有单组策略命中率与随机基线无统计显著差异(p>0.05)。排列3/5复式模式因纯数学覆盖优势(每位置3候选)大幅领先。

## 项目结构

```
bocai/
├── app.py                 # Flask入口
├── config.py              # 全局配置
├── lottery/
│   ├── analyzer.py        # 统计引擎（频率/遗漏/趋势）
│   ├── recommender.py     # 推荐引擎（6组+复式）
│   ├── backtester.py      # v3回测引擎
│   ├── detailed_backtest.py  # 精细化全管线回测
│   ├── ml_scorer.py       # ML评分（GBM/RF/LR）
│   ├── stacking.py        # Stacking集成
│   ├── adaptive.py        # 自适应权重
│   ├── bandit.py          # Thompson采样
│   ├── arima_predictor.py # AR(1)预测
│   ├── conditional.py     # 条件概率+反冷号
│   ├── features.py        # 特征工程+间距+星期几
│   ├── randomness.py      # 随机性检测
│   ├── ev_calc.py         # 期望值+覆盖率+贪心覆盖
│   ├── hyperopt.py        # 超参网格搜索
│   ├── fetcher.py         # 数据抓取
│   └── models.py          # SQLite ORM
├── templates/             # Jinja2模板
└── data/                  # 数据库
```
