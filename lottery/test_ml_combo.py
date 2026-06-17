# -*- coding: utf-8 -*-
"""测试 ML 驱动复式覆盖方案"""
import sys; sys.path.insert(0, ".")
from lottery import models, ml_scorer; models.init_db()
from lottery.analyzer import _per_number_stats, _build_window
from lottery.recommender import _score_map
import random as rnd

def test_ml_combo(game, n_max=200, step=3):
    cfg_map = {
        "pl3": {"positions": 3, "digit_pool": list(range(10))},
        "pl5": {"positions": 5, "digit_pool": list(range(10))},
        "dlt": {"type": "lotto", "front_pool": list(range(1, 36)), "back_pool": list(range(1, 13))},
    }
    cfg = cfg_map[game]
    all_recs = models.fetch_draws(game, limit=3000, order_desc=True)
    
    ml_h = trad_h = rand_h = 0
    ml_p = trad_p = rand_p = 1
    n_pts = 0
    
    is_lotto = cfg.get("type") == "lotto"
    
    for i in range(1, min(n_max, len(all_recs) - 30), step):
        test_rec = all_recs[i]
        train = all_recs[i + 1: i + 800]
        if len(train) < 30:
            continue
        win = _build_window(train, 730)
        if len(win) < 20:
            win = train
        
        if is_lotto:
            actual_front = set(test_rec["front"])
            pool = cfg["front_pool"]
            front_sets = [set(r["front"]) for r in win]
            stats = _per_number_stats(pool, front_sets, len(win), 30, 50)
            
            ml = ml_scorer.MLScorer(game, model_type="gbm")
            ml.fit(train)
            ml_score = ml.predict(stats, pool, len(win))
            ml_top3 = sorted(ml_score, key=lambda n: -ml_score.get(n, 0))[:3]
            ml_h += len(set(ml_top3) & actual_front)
            ml_p += 3
            
            trad_score = _score_map(stats)
            trad_top3 = sorted(trad_score, key=lambda n: -trad_score.get(n, 0))[:3]
            trad_h += len(set(trad_top3) & actual_front)
            trad_p += 3
            
            rand_top3 = rnd.sample(pool, 3)
            rand_h += len(set(rand_top3) & actual_front)
            rand_p += 3
        else:
            positions = cfg["positions"]
            pool = cfg["digit_pool"]
            actual = list(test_rec["front"])
            
            ml = ml_scorer.MLScorer(game, model_type="gbm")
            ml.fit(train)
            
            for pos in range(positions):
                seq = []
                for r in win:
                    digits = r["front"]
                    seq.append({digits[pos]} if pos < len(digits) else set())
                stats = _per_number_stats(pool, seq, len(win), 30, 50)
                
                s1 = ml.predict(stats, pool, len(win))
                t1 = sorted(s1, key=lambda n: -s1.get(n, 0))[:2]
                ml_h += 1 if actual[pos] in t1 else 0
                ml_p += 1
                
                s2 = _score_map(stats)
                t2 = sorted(s2, key=lambda n: -s2.get(n, 0))[:2]
                trad_h += 1 if actual[pos] in t2 else 0
                trad_p += 1
                
                t3 = rnd.sample(pool, 2)
                rand_h += 1 if actual[pos] in t3 else 0
                rand_p += 1
        
        n_pts += 1
    
    return {
        "ml_rate": ml_h / ml_p, "trad_rate": trad_h / trad_p, "rand_rate": rand_h / rand_p,
        "ml_hits": ml_h, "trad_hits": trad_h, "rand_hits": rand_h,
        "n_pts": n_pts,
    }

if __name__ == "__main__":
    for g in ["dlt", "pl3", "pl5"]:
        r = test_ml_combo(g, n_max=200, step=5)
        print(f"{g}: ML={r['ml_rate']:.2%} Trad={r['trad_rate']:.2%} Rand={r['rand_rate']:.2%} pts={r['n_pts']}")
