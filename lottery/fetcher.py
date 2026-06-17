# -*- coding: utf-8 -*-
"""数据抓取模块 — 从 500.com datachart 获取排列3/排列5/大乐透历史开奖。

数据源布局（经实测，编码 gb18030）：
  大乐透  /dlt/history/newinc/history.php?limit=N
     行 = [期号, 前1,前2,前3,前4,前5, 后1,后2, 销售额, 一等注,一等金, 二等注,二等金, 奖池, 日期]
     (第 0 行为表头，跳过)
  排列5  /plw/history/inc/history.php?limit=N
     行 = [期号, "d1 d2 d3 d4 d5"(空格分隔), 和值, 销售额, 日期]
  排列3  /pls/history/inc/history.php?limit=N
     行 = [期号, "d1 d2 d3"(空格分隔), 和值, 销售额, ...奖金列..., 日期]
  日期恒为最后一个 td。
"""

import logging
import re
import time

import requests
from bs4 import BeautifulSoup

from config import (
    REQUEST_HEADERS, REQUEST_TIMEOUT, REQUEST_SLEEP, REQUEST_RETRIES,
    INITIAL_FETCH_LIMIT, GAMES,
)
from . import models

logger = logging.getLogger(__name__)

# 各玩法的 inc/newinc 路径（已实测）
FETCH_PATHS = {
    "dlt": "/dlt/history/newinc/history.php",
    "pl3": "/pls/history/inc/history.php",
    "pl5": "/plw/history/inc/history.php",
}
BASE = "https://datachart.500.com"
ENCODING = "gb18030"

# 数据校验规则
VALIDATION_RULES = {
    "dlt": {
        "front_range": (1, 35),    # 前区号码范围
        "back_range": (1, 12),     # 后区号码范围
        "front_pick": 5,           # 前区选几个
        "back_pick": 2,            # 后区选几个
    },
    "pl3": {
        "digit_range": (0, 9),     # 每位数字范围
        "positions": 3,            # 几位
    },
    "pl5": {
        "digit_range": (0, 9),
        "positions": 5,
    },
}

# 日期格式正则
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_row(game: str, issue: str, draw_date: str,
                  front: list, back: list) -> list:
    """校验一行解析后的数据，返回错误信息列表（空列表=通过）。

    校验项：
      1. 号码范围检查（lotto: 前区/后区; digit: 各位数字）
      2. 重复号检查（lotto 前区/后区不应有重复）
      3. 号码数量检查（应与规则一致）
      4. 日期格式检查
      5. 期号格式检查
    """
    errors = []
    rules = VALIDATION_RULES.get(game, {})

    # 期号检查
    if not issue or not issue.isdigit():
        errors.append(f"期号格式异常: '{issue}'")

    # 日期格式检查
    if draw_date and not DATE_RE.match(draw_date):
        errors.append(f"日期格式异常: '{draw_date}'")

    if game == "dlt":
        # 号码数量
        if len(front) != rules["front_pick"]:
            errors.append(f"前区号码数={len(front)}，期望{rules['front_pick']}")
        if len(back) != rules["back_pick"]:
            errors.append(f"后区号码数={len(back)}，期望{rules['back_pick']}")

        # 范围检查
        lo_f, hi_f = rules["front_range"]
        for n in front:
            if not (lo_f <= n <= hi_f):
                errors.append(f"前区号码{n}超出范围[{lo_f},{hi_f}]")
        lo_b, hi_b = rules["back_range"]
        for n in back:
            if not (lo_b <= n <= hi_b):
                errors.append(f"后区号码{n}超出范围[{lo_b},{hi_b}]")

        # 重复检查
        if len(set(front)) != len(front):
            errors.append(f"前区有重复号码: {front}")
        if len(set(back)) != len(back):
            errors.append(f"后区有重复号码: {back}")
    else:
        # digit 玩法
        lo, hi = rules["digit_range"]
        expected_pos = rules["positions"]
        if len(front) != expected_pos:
            errors.append(f"位数={len(front)}，期望{expected_pos}")
        for i, n in enumerate(front):
            if not (lo <= n <= hi):
                errors.append(f"第{i+1}位数字{n}超出范围[{lo},{hi}]")

    return errors


def _fetch_html(game: str, limit: int) -> str:
    """请求某玩法历史数据 HTML，带重试和备选源降级。

    优先使用 datachart.500.com，失败后自动降级到 kaijiang.500.com。
    """
    from config import FALLBACK_BASE, FALLBACK_PATHS, FALLBACK_NAME

    # 主源和备选源列表
    sources = [
        (BASE, FETCH_PATHS[game], "主源"),
        (FALLBACK_BASE, FALLBACK_PATHS.get(game, FETCH_PATHS[game]), "备选源"),
    ]

    last_err = None
    for base_url, path, source_label in sources:
        url = base_url + path + f"?limit={limit}"
        logger.info(f"[{game}] 尝试{source_label}: {url[:60]}...")
        for attempt in range(1, REQUEST_RETRIES + 1):
            try:
                headers = dict(REQUEST_HEADERS)
                headers["Referer"] = base_url + "/"
                resp = requests.get(
                    url, headers=headers, timeout=REQUEST_TIMEOUT
                )
                resp.encoding = ENCODING
                if resp.status_code == 200 and resp.text.strip():
                    logger.info(f"[{game}] {source_label}成功（第{attempt}次尝试）")
                    return resp.text
                last_err = f"HTTP {resp.status_code}"
            except requests.RequestException as e:
                last_err = str(e)
            time.sleep(REQUEST_SLEEP)
        logger.warning(f"[{game}] {source_label}({base_url})全部重试失败: {last_err}")

    raise RuntimeError(f"抓取 {game} 失败（主源+备选源均不可用）: {last_err}")


def _parse_rows(game: str, html: str):
    """解析 HTML，返回 [(issue, draw_date, front, back, numbers)] 列表，最新在前。
    
    带数据校验：跳过校验失败的行并记录警告。
    """
    soup = BeautifulSoup(html, "lxml")
    rows = soup.select("tr.t_tr1")
    out = []
    skipped = 0
    for tr in rows:
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not tds:
            continue
        issue = tds[0]
        # 跳过表头行（如 "1","2",... 或 "期号"）
        if not issue.isdigit() or len(issue) < 4:
            continue
        draw_date = tds[-1]
        # 规范化日期：取前 10 位 YYYY-MM-DD
        if len(draw_date) >= 10:
            draw_date = draw_date[:10]
        else:
            draw_date = ""

        try:
            if game == "dlt":
                # 前 5 = 前区, 6-7 = 后区；其余为奖金/奖池
                if len(tds) < 8:
                    skipped += 1
                    continue
                front = [int(tds[i]) for i in range(1, 6)]
                back = [int(tds[i]) for i in range(6, 8)]
                numbers = front + back
            else:
                # pl3 / pl5：第 1 个 td 为空格分隔的各位数字
                if len(tds) < 2:
                    skipped += 1
                    continue
                digits = [int(x) for x in tds[1].split()]
                front = digits
                back = []
                numbers = digits
        except (ValueError, IndexError) as e:
            logger.warning(f"[{game}] 期号{issue}解析失败: {e}")
            skipped += 1
            continue

        # 数据校验
        errors = _validate_row(game, issue, draw_date, front, back)
        if errors:
            logger.warning(f"[{game}] 期号{issue}校验失败，跳过: {'; '.join(errors)}")
            skipped += 1
            continue

        out.append((issue, draw_date, front, back, numbers))

    if skipped > 0:
        logger.info(f"[{game}] 解析完成: 通过 {len(out)} 条，跳过 {skipped} 条")
    return out


def fetch_and_store(game: str, limit: int = None, log: bool = True) -> int:
    """抓取某玩法并增量入库，返回新增条数。
    
    含数据校验：范围检查、去重检查、期号递增检查。
    """
    if limit is None:
        limit = INITIAL_FETCH_LIMIT
    try:
        html = _fetch_html(game, limit)
        records = _parse_rows(game, html)
    except Exception as e:
        if log:
            models.log_refresh(game, 0, "error", str(e))
        raise

    # 期号递增检查：解析后数据按时间应有序，检测大范围乱序
    if len(records) > 1:
        issues = [r[0] for r in records]
        # 500.com 返回最新在前，所以期号应大致递减
        disorder_count = 0
        for i in range(1, min(len(issues), 200)):
            if issues[i] > issues[i - 1]:
                disorder_count += 1
        if disorder_count > len(issues) * 0.1:
            logger.warning(f"[{game}] 期号顺序异常：{disorder_count}/{len(issues)} 处逆序")

    new_count = 0
    dup_in_batch = 0
    seen_issues = set()
    for issue, draw_date, front, back, numbers in records:
        # 批次内去重（防止同一次抓取中有重复期号）
        if issue in seen_issues:
            dup_in_batch += 1
            continue
        seen_issues.add(issue)
        if models.upsert_draw(game, issue, draw_date, numbers, front, back):
            new_count += 1

    msg = f"解析 {len(records)} 条，新增 {new_count} 条"
    if dup_in_batch > 0:
        msg += f"，批次内重复 {dup_in_batch} 条已跳过"
    if log:
        models.log_refresh(game, new_count, "ok", msg)
    return new_count


def refresh_all(limit: int = None):
    """抓取全部三个玩法，返回 {game: new_count}。失败的游戏返回 ('error', msg)。"""
    results = {}
    for game in ("dlt", "pl3", "pl5"):
        try:
            results[game] = fetch_and_store(game, limit=limit)
        except Exception as e:
            results[game] = ("error", str(e))
        time.sleep(REQUEST_SLEEP)
    return results


if __name__ == "__main__":
    models.init_db()
    print("开始抓取全部玩法（首次将获取大量历史数据，请耐心等待）...")
    res = refresh_all()
    for g, v in res.items():
        if isinstance(v, tuple):
            print(f"  {g}: 失败 - {v[1]}")
        else:
            print(f"  {g}: 新增 {v} 条，库中共 {models.count_draws(g)} 条")
