#!/usr/bin/env python3
"""
Zero-setup generator for a realistic, analysis-ready `tasks.csv` **with progress logs**.

• No CLI flags. Place this file next to your trips CSV (and, if available,
  `hourly_demand_by_vendor.csv`, `incidents.csv`, `status_changes.csv`,
  `vehicles.csv`, `service_areas.csv`). Then run:

    python3 generate_tasks_nointput.py

• Calibrates to your actual demand and upstream events to avoid flat outputs.
• Emits rich fields so you can build competitive visuals without extra DAX.
• Verbose progress logs show loading, linking, row progress, and summaries.

Task types generated
--------------------
- battery_swap            (often created from `status_changes[battery_low]`)
- rebalance               (often created from `status_changes[rebalance]`)
- pickup_illegal_parking  (often created from `incidents[illegal_parking]`)
- repair                  (often created from incidents like `crash`/`vandalism`)

Output columns
--------------
 task_id, vendor, task_type, task_subtype, priority, status,
 created_at, assigned_at, completed_at,
 response_minutes, work_minutes, total_minutes,
 sla_target_min, sla_met_int, deadline_at,
 service_area_id, community_area_name, area_type, hour_band, weekday,
 assigned_to, team, source, linked_incident_id, linked_event_id,
 cost_usd, parts_cost_usd, workload_trips_hour,
 day, hour

Join keys: day/hour/vendor/service_area_id ↔ hourly_demand_by_vendor; vendor ↔ vehicles.

Dependencies: Python 3.9+, pandas, numpy (pyarrow optional)
"""

from __future__ import annotations
import logging
import math
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd

# ---------------- config ----------------
SEED = 46
NB_K = 7.0  # negbin dispersion for hourly counts

# How often to emit progress logs (rows in the hourly table)
PROGRESS_EVERY_ROWS = 50_000

# SLA targets (minutes)
SLA_TARGET = {
    "battery_swap": 120,
    "rebalance": 180,
    "pickup_illegal_parking": 90,
    "repair": 1440,  # 24h
}

# Baseline task creation intensities
PER_1K_TRIPS = {
    "battery_swap": 1.2,  # per 1k trips
    "rebalance": 1.4,
    "pickup_illegal_parking": 0.9,
    "repair": 0.35,
}

# Modifiers
HOUR_BAND_MULT = {
    "battery_swap": {"late_night":0.7, "morning":1.0, "midday":1.05, "evening":1.15, "night":1.1},
    "rebalance":   {"late_night":0.9, "morning":1.25, "midday":1.0,  "evening":1.2,  "night":0.9},
    "pickup_illegal_parking": {"late_night":1.25,"morning":0.85,"midday":1.0,"evening":1.15,"night":1.35},
    "repair":      {"late_night":0.8, "morning":1.0,  "midday":1.05, "evening":1.1,  "night":0.95},
}
AREA_MULT = {
    "battery_swap": {"downtown":1.05, "campus":1.0,  "industrial":0.95, "residential":1.0},
    "rebalance":   {"downtown":1.3,  "campus":1.15, "industrial":0.9,  "residential":1.0},
    "pickup_illegal_parking": {"downtown":1.35, "campus":1.25, "industrial":0.8, "residential":1.0},
    "repair":      {"downtown":1.1,  "campus":1.05, "industrial":1.1, "residential":1.0},
}
VENDOR_MULT = {"Lyft":1.0, "Lime":0.98, "Spin":1.05, "Coco":1.08}

# Duration models (lognormal parameters) — response (create→assign) and work (assign→complete)
RESP_LOGN = {
    "battery_swap": (math.log(40), 0.55),
    "rebalance":    (math.log(55), 0.55),
    "pickup_illegal_parking": (math.log(35), 0.5),
    "repair":       (math.log(240),0.7),
}
WORK_LOGN = {
    "battery_swap": (math.log(45), 0.5),
    "rebalance":    (math.log(60), 0.55),
    "pickup_illegal_parking": (math.log(40), 0.5),
    "repair":       (math.log(360),0.7),
}

TEAM_LIST = ["tech_01","tech_02","tech_03","tech_04"]
ASSIGNEES = ["tech_01","tech_02","tech_03","tech_04","tech_05","tech_06"]
SOURCES = ["incident_link","sensor_alert","field_op","patrol","citizen_report"]
REB_SUBTYPES = ["move_to_hotspot","clear_footpath","event_coverage"]

# ---------------- file discovery ----------------
TRIPS_FILE_CANDIDATES = ["e_scooter_trips.csv","e-scooter-trips.csv","chicago_trips.csv","trips.csv"]
HOURLY_FILE_CANDIDATES = ["hourly_demand_by_vendor.csv","mds_export/hourly_demand_by_vendor.csv"]
SERVICE_AREAS_FILE_CANDIDATES = ["service_areas.csv","mds_export/service_areas.csv","../mds_export/service_areas.csv"]
VEHICLES_FILE_CANDIDATES = ["vehicles.csv","mds_export/vehicles.csv","../mds_export/vehicles.csv"]
INCIDENTS_FILE_CANDIDATES = ["incidents.csv","mds_export/incidents.csv","../mds_export/incidents.csv"]
STATUS_FILE_CANDIDATES = ["status_changes.csv","mds_export/status_changes.csv","../mds_export/status_changes.csv"]

LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("tasks_auto")

VENDOR_MAP = {"lyft":"Lyft","lime":"Lime","spin":"Spin","coco":"Coco","bird":"Coco","tier":"Coco","superpedestrian":"Spin"}

# ---------------- helpers ----------------

def norm(s: str) -> str:
    return "" if s is None else "".join(ch for ch in str(s).lower() if ch.isalnum())

def canon_vendor(x: object) -> str:
    key = norm("" if pd.isna(x) else x)
    return VENDOR_MAP.get(key, "Coco")


def find_existing(path: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = (path / name).resolve()
        if p.exists():
            return p
    for p in path.glob("*.csv"):
        try:
            head = pd.read_csv(p, nrows=0)
            cols = [c.lower() for c in head.columns.astype(str)]
            if "vendor" in cols:
                return p
        except Exception:
            continue
    return None


def hour_band(h: int) -> str:
    if 0 <= h <= 5:
        return "late_night"
    elif 6 <= h <= 9:
        return "morning"
    elif 10 <= h <= 15:
        return "midday"
    elif 16 <= h <= 19:
        return "evening"
    else:
        return "night"

# ---------------- loaders ----------------

def load_hourly_and_meta(here: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trips_path = find_existing(here, TRIPS_FILE_CANDIDATES)
    if not trips_path:
        raise SystemExit("Trips CSV not found")
    hourly_path = find_existing(here, HOURLY_FILE_CANDIDATES)
    svc_path = find_existing(here, SERVICE_AREAS_FILE_CANDIDATES)

    if hourly_path:
        hourly = pd.read_csv(hourly_path)
        hourly["day"] = pd.to_datetime(hourly["day"], errors="coerce").dt.date
        hourly["hour"] = pd.to_numeric(hourly["hour"], errors="coerce").astype(int)
        logger.info("Loaded hourly_demand_by_vendor.csv (%d rows)", len(hourly))
    else:
        # Derive a minimal hourly table from trips
        head = pd.read_csv(trips_path, nrows=0)
        cols = list(head.columns)
        ncols = {c: norm(c) for c in cols}
        def match(cands):
            cands_n = [norm(c) for c in cands]
            for c, nc in ncols.items():
                if nc in cands_n: return c
            for c, nc in ncols.items():
                if any(nc.find(x) >= 0 for x in cands_n): return c
            return None
        st = match(["start_time","started_at","trip_start","startdatetime","starttimestamp","startdate","starttime"])
        vendor_col = match(["vendor","provider","provider_name","company","operator"])
        sname = match(["start_community_area_name","start_area_name","start_neighborhood","start_zone_name","start_community_area"])
        ename = match(["end_community_area_name","end_area_name","end_neighborhood","end_zone_name","end_community_area"])
        snum = match(["start_community_area_number","start_area_id","start_area_code"])
        enum = match(["end_community_area_number","end_area_id","end_area_code"])
        use = [c for c in (st,vendor_col,sname,ename,snum,enum) if c]
        df = pd.read_csv(trips_path, usecols=use)
        df[st] = pd.to_datetime(df[st], errors="coerce")
        df = df.dropna(subset=[st])
        df["vendor"] = df[vendor_col].map(canon_vendor)
        df["day"] = df[st].dt.date
        df["hour"] = df[st].dt.hour.astype(int)
        if sname or ename:
            area = pd.Series([""] * len(df), index=df.index, dtype="object")
            if sname is not None:
                area = df[sname].astype(str).str.strip()
            if ename is not None:
                area = area.where(area.notna() & (area.astype(str).str.len() > 0), df[ename].astype(str).str.strip())
            df["community_area_name"] = area.fillna("").astype(str).str.strip()
            names = sorted(df["community_area_name"].dropna().astype(str).str.lower().unique())
            name_to_id = {nm: i+1 for i, nm in enumerate(names)}
            df["service_area_id"] = df["community_area_name"].str.lower().map(name_to_id)
        else:
            nums = pd.to_numeric(df[snum] if snum else df[enum], errors="coerce")
            df["service_area_id"] = nums.astype("Int64").astype(float).astype("Int64")
            df["community_area_name"] = df["service_area_id"].apply(lambda x: f"CA {int(x):02d}" if pd.notna(x) else "")
        hourly = df.groupby(["day","hour","service_area_id","community_area_name","vendor"], as_index=False).size()
        hourly = hourly.rename(columns={"size":"trips"})
        logger.info("Derived hourly demand from trips (%d rows)", len(hourly))

    if svc_path:
        svc = pd.read_csv(svc_path)
        if "area_type" not in svc.columns:
            svc["area_type"] = "residential"
        svc = svc[["service_area_id","community_area_name","area_type"]]
        logger.info("Loaded service_areas.csv (%d areas)", len(svc))
    else:
        svc = hourly[["service_area_id","community_area_name"]].drop_duplicates()
        svc["area_type"] = "residential"
        logger.info("No service_areas.csv found; inferred %d areas from hourly", len(svc))

    hourly = hourly.merge(svc, on=["service_area_id","community_area_name"], how="left")
    return hourly, svc


def load_optional(here: Path):
    paths = {
        "incidents": find_existing(here, INCIDENTS_FILE_CANDIDATES),
        "status": find_existing(here, STATUS_FILE_CANDIDATES),
        "vehicles": find_existing(here, VEHICLES_FILE_CANDIDATES),
    }
    frames = {}
    for k,p in paths.items():
        if p is None:
            frames[k] = None
            logger.info("Optional file not found: %s", k)
            continue
        try:
            df = pd.read_csv(p)
            frames[k] = df
            logger.info("Loaded optional %s (%d rows) from %s", k, len(df), p)
        except Exception as e:
            frames[k] = None
            logger.warning("Failed to read optional %s: %s", k, e)
    return frames

# ---------------- generation ----------------

def negbin(mean: float, k: float, rng: np.random.Generator) -> int:
    if mean <= 0: return 0
    lam = rng.gamma(shape=k, scale=mean/max(k,1e-6))
    return int(rng.poisson(lam))


def add_time_noise(dt: datetime, rr: random.Random) -> datetime:
    return dt + timedelta(minutes=rr.randint(0,59), seconds=rr.randint(0,59))


def draw_lognormal(mu_sigma: Tuple[float,float], adj: float=1.0) -> int:
    mu,sigma = mu_sigma
    return max(1, int(np.random.lognormal(mean=mu, sigma=sigma) * adj))


def priority_for(task_type: str, area_type: str, band: str) -> int:
    # 1=highest, 3=lowest
    base = {"battery_swap":2, "rebalance":2, "pickup_illegal_parking":1, "repair":2}.get(task_type,2)
    if task_type=="pickup_illegal_parking" and area_type=="downtown": base = 1
    if task_type=="repair" and band in ("evening","night"): base = 1
    return base


def cost_model(task_type: str, move_km: float) -> Tuple[float,float]:
    # returns (ops cost, parts cost)
    if task_type=="battery_swap":
        return (6 + 0.8*move_km, 0)
    if task_type=="rebalance":
        return (5 + 0.6*move_km, 0)
    if task_type=="pickup_illegal_parking":
        return (7 + 1.0*move_km, 0)
    if task_type=="repair":
        return (8 + 0.5*move_km, 12 + 20*np.random.random())
    return (3,0)


def generate_tasks() -> pd.DataFrame:
    start_time = time.time()
    rng = np.random.default_rng(SEED)
    rr = random.Random(SEED)
    here = Path(__file__).resolve().parent

    hourly, svc = load_hourly_and_meta(here)
    opt = load_optional(here)
    incidents = opt["incidents"]
    status = opt["status"]

    # Summary of scope
    days = sorted(hourly["day"].unique())
    vendors = sorted(hourly["vendor"].unique())
    areas = hourly["service_area_id"].nunique()
    logger.info("Scope: %d days, %d vendors, %d areas, %d hourly rows", len(days), len(vendors), areas, len(hourly))

    records: List[Dict[str,object]] = []
    tid = 1

    area_type_map = svc.set_index("service_area_id")["area_type"].to_dict()

    # Daily noise and backlog pressure per vendor
    day_noise = {d: rng.gamma(shape=6.0, scale=1.0/6.0) for d in days}
    backlog_factor = {v: 1.0 for v in vendors}

    # 1) Tasks sourced from linkable rows (incidents/status)
    link_created = 0
    if incidents is not None or status is not None:
        logger.info("Pass 1/2: Linking tasks from incidents/status…")
    linkable_rows = []
    if incidents is not None:
        try:
            inc = incidents.copy()
            inc["occurred_at"] = pd.to_datetime(inc["occurred_at"], errors="coerce")
            inc = inc.dropna(subset=["occurred_at"])  # keep valid rows
            linkable_rows.append(("illegal_parking", inc[inc["incident_type"]=="illegal_parking"]))
            linkable_rows.append(("repair", inc[inc["incident_type"].isin(["crash","vandalism"]) ]))
            logger.info("  Loaded incidents for linking: %d rows", len(inc))
        except Exception as e:
            logger.warning("  Could not use incidents: %s", e)

    if status is not None:
        try:
            st = status.copy()
            st["occurred_at"] = pd.to_datetime(st["occurred_at"], errors="coerce")
            st = st.dropna(subset=["occurred_at"])  # keep valid rows
            linkable_rows.append(("battery_low", st[st["event_type"]=="battery_low"]))
            linkable_rows.append(("rebalance", st[st["event_type"]=="rebalance"]))
            logger.info("  Loaded status_changes for linking: %d rows", len(st))
        except Exception as e:
            logger.warning("  Could not use status_changes: %s", e)

    def push_task(vendor: str, aid: int, can: str, created: datetime, task_type: str,
                  subtype: Optional[str], source: str,
                  linked_incident_id: Optional[int], linked_event_id: Optional[int],
                  trips_hour: float):
        nonlocal tid, link_created
        atype = area_type_map.get(int(aid), "residential")
        band = hour_band(created.hour)
        # response/work mins with load/backlog adjustments
        resp_adj = 1.0 * (1.0 + 0.15*(trips_hour/200.0)) * backlog_factor.get(vendor,1.0)
        work_adj = 1.0 * (1.0 + 0.10*(trips_hour/200.0))
        # area/band tweak
        if atype=="downtown":
            resp_adj *= 1.10
        if band in ("evening","night"):
            resp_adj *= 1.12

        response = draw_lognormal(RESP_LOGN[task_type], resp_adj)
        work = draw_lognormal(WORK_LOGN[task_type], work_adj)

        assigned_at = created + timedelta(minutes=response)
        completed_at = assigned_at + timedelta(minutes=work)
        total = response + work

        # move distance proxy from workload
        move_km = max(0.1, float(np.random.lognormal(math.log(2.0), 0.5)))
        cost_ops, cost_parts = cost_model(task_type, move_km)

        # SLA
        sla = SLA_TARGET[task_type]
        sla_met = 1 if total <= sla else 0
        deadline_at = created + timedelta(minutes=sla)

        # probability some tasks remain open/assigned (10–15%)
        status_choice = random.Random(SEED + tid).choices(["done","open","assigned","canceled"], weights=[0.78,0.08,0.10,0.04])[0]
        status_val = status_choice
        assigned_out = assigned_at if status_val in ("assigned","done","canceled") else ""
        completed_out = completed_at if status_val=="done" else ""
        if status_val=="canceled":
            completed_out = assigned_at + timedelta(minutes=random.Random(SEED + tid).randint(5,30))

        prio = priority_for(task_type, atype, band)
        assignee = random.Random(SEED + tid).choice(ASSIGNEES)
        team = random.Random(SEED + tid).choice(TEAM_LIST)

        records.append({
            "task_id": tid,
            "vendor": vendor,
            "task_type": task_type,
            "task_subtype": subtype or "",
            "priority": prio,
            "status": status_val,
            "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
            "assigned_at": assigned_out.strftime("%Y-%m-%d %H:%M:%S") if assigned_out else "",
            "completed_at": completed_out.strftime("%Y-%m-%d %H:%M:%S") if completed_out else "",
            "response_minutes": response if status_val in ("assigned","done","canceled") else None,
            "work_minutes": work if status_val=="done" else None,
            "total_minutes": total if status_val=="done" else None,
            "sla_target_min": sla,
            "sla_met_int": sla_met if status_val=="done" else None,
            "deadline_at": deadline_at.strftime("%Y-%m-%d %H:%M:%S"),
            "service_area_id": int(aid),
            "community_area_name": can,
            "area_type": atype,
            "hour_band": band,
            "weekday": int(created.weekday()),
            "assigned_to": assignee,
            "team": team,
            "source": source,
            "linked_incident_id": int(linked_incident_id) if linked_incident_id is not None else "",
            "linked_event_id": int(linked_event_id) if linked_event_id is not None else "",
            "cost_usd": round(cost_ops,2),
            "parts_cost_usd": round(cost_parts,2),
            "workload_trips_hour": float(trips_hour),
            "day": created.date(),
            "hour": created.hour,
        })
        tid += 1
        if source == "incident_link":
            link_created += 1

    # Pass 1: link from upstream rows
    link_counter = {"battery_swap":0, "rebalance":0, "pickup_illegal_parking":0, "repair":0}
    for kind, df in linkable_rows:
        if df is None or df.empty:
            continue
        if kind=="illegal_parking":
            task_type = "pickup_illegal_parking"; prob = 0.9
            link_id_col = "incident_id"; event_id_col = None
        elif kind=="repair":
            task_type = "repair"; prob = 0.45
            link_id_col = "incident_id"; event_id_col = None
        elif kind=="battery_low":
            task_type = "battery_swap"; prob = 0.65
            link_id_col = None; event_id_col = "event_id"
        elif kind=="rebalance":
            task_type = "rebalance"; prob = 0.70
            link_id_col = None; event_id_col = "event_id"
        else:
            continue
        total_rows = len(df)
        kept = 0
        for i, row in enumerate(df.itertuples(index=False), 1):
            if random.Random(SEED + i).random() > prob:
                continue
            vendor = str(getattr(row, "vendor", "Coco"))
            created = pd.to_datetime(getattr(row, "occurred_at"), errors="coerce")
            if pd.isna(created):
                continue
            created = add_time_noise(created.to_pydatetime(), random.Random(SEED + i))
            aid = int(getattr(row, "service_area_id", 1))
            can = str(getattr(row, "community_area_name", f"Area {aid}"))
            subset = hourly[(hourly["day"]==created.date()) & (hourly["hour"]==created.hour) &
                            (hourly["service_area_id"]==aid) & (hourly["vendor"]==vendor)]
            trips_hour = float(subset["trips"].sum()) if not subset.empty else 0.0
            subtype = random.Random(SEED + i).choice(REB_SUBTYPES) if task_type=="rebalance" else ""
            l_inc = int(getattr(row, "incident_id", 0)) if link_id_col=="incident_id" else None
            l_evt = int(getattr(row, "event_id", 0)) if event_id_col=="event_id" else None
            source = "incident_link" if (l_inc or l_evt) else "sensor_alert"
            push_task(vendor, aid, can, created, task_type, subtype, source, l_inc, l_evt, trips_hour)
            kept += 1
            if kept and kept % 10000 == 0:
                logger.info("  Linked %s: %d/%d (%.1f%%)", task_type, kept, total_rows, 100*kept/total_rows)
        link_counter[task_type] += kept
        logger.info("  Linked %s tasks: %d (from %d upstream rows)", task_type, kept, total_rows)

    # 2) Proactive tasks drawn from hourly demand (fills gaps, keeps dynamics)
    logger.info("Pass 2/2: Generating proactive tasks from hourly demand…")
    total_rows = len(hourly)
    for idx, row in enumerate(hourly.itertuples(index=False), 1):
        d: date = row.day
        h: int = int(row.hour)
        vendor: str = str(row.vendor)
        trips: float = float(row.trips)
        aid: int = int(row.service_area_id)
        can: str = str(row.community_area_name)
        atype: str = str(row.area_type).lower() if hasattr(row, 'area_type') else 'residential'

        band = hour_band(h)
        dn = day_noise[d]

        for task_type in ("battery_swap","rebalance","pickup_illegal_parking","repair"):
            lam = (trips * (PER_1K_TRIPS[task_type] / 1000.0)
                   * HOUR_BAND_MULT[task_type][band]
                   * AREA_MULT[task_type].get(atype, 1.0)
                   * VENDOR_MULT.get(vendor, 1.0) * dn)
            lam *= np.random.lognormal(0.0, 0.12)
            n = negbin(lam, NB_K, np.random.default_rng(SEED + idx))
            for _ in range(n):
                created = datetime(d.year,d.month,d.day,h, random.Random(SEED + tid).randint(0,59), random.Random(SEED + tid).randint(0,59))
                subtype = random.Random(SEED + tid).choice(REB_SUBTYPES) if task_type=="rebalance" else ""
                push_task(vendor, aid, can, created, task_type, subtype, "sensor_alert", None, None, trips)

        if idx % PROGRESS_EVERY_ROWS == 0:
            elapsed = time.time() - start_time
            logger.info("  Progress: %d/%d hourly rows (%.1f%%), tasks so far=%d, elapsed=%.1fs",
                        idx, total_rows, 100*idx/total_rows, tid-1, elapsed)

    df = pd.DataFrame.from_records(records)

    # End-of-run summaries in logs
    if not df.empty:
        try:
            by_type = df["task_type"].value_counts().to_dict()
            by_vendor = df.groupby("vendor").size().to_dict()
            sla = df[df["sla_met_int"].notna()].groupby("vendor")["sla_met_int"].mean().round(3).to_dict()
            logger.info("Summary: tasks by type = %s", by_type)
            logger.info("Summary: tasks by vendor = %s", by_vendor)
            logger.info("Summary: SLA %% by vendor = %s", sla)
        except Exception:
            pass

    total_elapsed = time.time() - start_time
    logger.info("Task generation complete in %.1fs (rows=%d, tasks=%d)", total_elapsed, len(hourly), len(df))

    return df

# ---------------- main ----------------

def main():
    df = generate_tasks()
    here = Path(__file__).resolve().parent
    out_path = here / "tasks.csv"
    cols = [
        "task_id","vendor","task_type","task_subtype","priority","status",
        "created_at","assigned_at","completed_at",
        "response_minutes","work_minutes","total_minutes",
        "sla_target_min","sla_met_int","deadline_at",
        "service_area_id","community_area_name","area_type","hour_band","weekday",
        "assigned_to","team","source","linked_incident_id","linked_event_id",
        "cost_usd","parts_cost_usd","workload_trips_hour","day","hour"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d tasks to %s", len(df), out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        raise
