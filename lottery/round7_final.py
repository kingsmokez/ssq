# -*- coding: utf-8 -*-
"""第七轮最终全面对比报告"""
import sys; sys.path.insert(0, ".")
from lottery import models, detailed_backtest; models.init_db()

for g in ["dlt", "pl3", "pl5"]:
    print("")
    detailed_backtest.print_detailed_report(g, n_test=300, step=3)
    print("")

print("=" * 70)
print("  ROUND 7 COMPLETE")
print("=" * 70)
