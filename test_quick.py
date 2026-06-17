#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick test: run recommend() and verify output."""

from lottery.recommender import recommend
from lottery.analyzer import analyze
from lottery import models

models.init_db()

for game in ("dlt", "pl3", "pl5"):
    gs = analyze(game)
    result = recommend(game, gs)
    print(f"\n=== {gs.name} ({game}) ===")
    for g in result["single"]:
        print(f"  G{g['index']}: {g['label']} -> {g['picks']}")
    for mode in ("small", "medium"):
        c = result["combos"][mode]
        print(f"  Combo-{mode}: bets={c.get('bets',1)} cost={c.get('cost',2)}")

print("\nAll OK!")
