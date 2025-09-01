#!/usr/bin/env python3
"""
Zero-setup generator for a realistic, analysis-ready `status_changes.csv`.

• No CLI flags. Put this file next to your trips CSV (and, if available,
  `hourly_demand_by_vendor.csv`, `vehicles.csv`, `service_areas.csv`). Then run:

    python3 generate_status_changes_nointput.py

• Calibrates to your actual hourly demand and fleet to avoid flat outputs.
• Produces nuanced operational events with ready-to-plot columns so you don't
  need DAX for common KPIs (SLA booleans as 0/1, hour_band, weekday, costs, etc.).

Event types generated
---------------------
- battery_low      (has handling/SLA)
- rebalance        (has handling/SLA, with subtype reasons)
- offline          (short downtime, has handling/SLA)
- maintenance      (longer downtime, has handling/SLA)
- deployment       (morning push)  — informational
- retrieval        (late-night sweep) — informational

Output columns
--------------
 event_id, vendor, vehicle_id, event_type, event_subtype,
 occurred_at, handled_at, handle_minutes, sla_minutes, sla_met_int,
 service_area_id, community_area_name, area_type, hour_band, weekday,
 move_km_est, battery_pct_before, battery_pct_after,
 offline_minutes, ops_team, compliance_flag, cost_usd,
 workload_trips_hour, day, hour

Joins: day/hour/vendor/service_area_id ↔ hourly_demand_by_vendor; vendor/vehicle_id ↔ vehicles.

Dependencies: Python 3.9+, pandas, numpy (pyarrow optional)
"""

from __future__ import annotations
import logging
import math
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd

# ---------------- config ----------------
SEED = 45
NB_K = 7.5   # negbin dispersion (smaller => spikier)

# SLA targets by type (minutes). None => informational, no SLA.
SLA_MINUTES = {
    "battery_low": 120,
    "rebalance": 180,
    "offline": 60,
    "maintenance": 480,
}

# Per-type baseline intensity parameters (these scale with demand/vehicles below)
PER_1K_TRIPS = {
    "battery_low": 1.6,    # per 1k trips
    "rebalance":   1.3,
}
PER_VEHICLE_PER_HOUR = {
    "offline": 0.0022,     # per active vehicle per hour
    "maintenance": 0.00025
}
DEPLOY_PER_ACTIVE = 0.015   # morning
RETRIEVE_PER_ACTIVE = 0.020  # late night

# Multipliers
HOUR_BAND_MULT = {
    "battery_low": {"late_night":0.8, "morning":1.0, "midday":1.0, "evening":1.15, "night":1.1},
    "rebalance":   {"late_night":0.9, "morning":1.25, "midday":1.0, "evening":1.2,  "night":0.9},
    "offline":     {"late_night":1.15,"morning":1.0, "midday":1.0, "evening":1.1,  "night":1.2},
    "maintenance": {"late_night":0.8, "morning":1.0, "midday":1.1, "evening":1.0,  "night":0.9},
}
AREA_MULT = {
    "battery_low": {"downtown":1.05, "campus":1.0,  "industrial":0.95, "residential":1.0},
    "rebalance":   {"downtown":1.25, "campus":1.15, "industrial":0.9,  "residential":1.0},
    "offline":     {"downtown":1.1,  "campus":1.0,  "industrial":0.95, "residential":1.0},
    "maintenance": {"downtown":1.0,  "campus":1.0,  "industrial":1.1,  "residential":1.0},
}
VENDOR_MULT = {"Lyft":1.0, "Lime":0.98, "Spin":1.05, "Coco":1.08}

# Handling time distributions (lognormal parameters)
HANDLE_LOGN = {
    "battery_low": (math.log(90), 0.5),
    "rebalance":   (math.log(120),0.55),
    "offline":     (math.log(40), 0.6),
    "maintenance": (math.log(360),0.7),
}

REB_SUBTYPES = ["move_to_hotspot","clear_footpath","event_coverage"]
OPS_TEAMS = ["tech_01","tech_02","tech_03","tech_04"]

# ---------------- file discovery ----------------
TRIPS_FILE_CANDIDATES = ["e_scooter_trips.csv","e-scooter-trips.csv","chicago_trips.csv","trips.csv"]
HOURLY_FILE_CANDIDATES = ["hourly_demand_by_vendor.csv","mds_export/hourly_demand_by_vendor.csv"]
SERVICE_AREAS_FILE_CANDIDATES = ["service_areas.csv","mds_export/service_areas.csv","../mds_export/service_areas.csv"]
VEHICLES_FILE_CANDIDATES = ["vehicles.csv","mds_export/vehicles.csv","../mds_export/vehicles.csv"]

LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("status_auto")

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
    if h <= 5: return "late_night"
    if h <= 9: return "morning"
    if h <= 15: return "midday"
    if h <= 19: return "evening"
    return "night"

# ---------------- loaders ----------------

def load_trips_for_hourly(trips_path: Path) -> pd.DataFrame:
    head = pd.read_csv(trips_path, nrows=0)
    cols = list(head.columns)
    ncols = {c: norm(c) for c in cols}

    def match(cands: List[str]) -> Optional[str]:
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

    # area names
    if sname or ename:
        area = pd.Series([""] * len(df), index=df.index, dtype="object")
        if sname is not None:
            area = df[sname].astype(str).str.strip()
        if ename is not None:
            area = area.where(area.notna() & (area.astype(str).str.len() > 0), df[ename].astype(str).str.strip())
        df["community_area_name"] = area.fillna("").astype(str).str.strip()
        # make ids by enumerating names
        names = sorted(df["community_area_name"].dropna().astype(str).str.lower().unique())
        name_to_id = {nm: i+1 for i, nm in enumerate(names)}
        df["service_area_id"] = df["community_area_name"].str.lower().map(name_to_id)
    else:
        # numeric only
        nums = pd.to_numeric(df[snum] if snum else df[enum], errors="coerce")
        df["service_area_id"] = nums.astype("Int64").astype(float).astype("Int64")
        df["community_area_name"] = df["service_area_id"].apply(lambda x: f"CA {int(x):02d}" if pd.notna(x) else "")

    hourly = df.groupby(["day","hour","service_area_id","community_area_name","vendor"], as_index=False).size()
    hourly = hourly.rename(columns={"size":"trips"})
    return hourly


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
    else:
        hourly = load_trips_for_hourly(trips_path)

    if svc_path:
        svc = pd.read_csv(svc_path)
        if "area_type" not in svc.columns:
            svc["area_type"] = "residential"
        svc = svc[["service_area_id","community_area_name","area_type"]]
    else:
        svc = hourly[["service_area_id","community_area_name"]].drop_duplicates()
        svc["area_type"] = "residential"

    hourly = hourly.merge(svc, on=["service_area_id","community_area_name"], how="left")
    return hourly, svc


def load_vehicles(here: Path) -> Optional[pd.DataFrame]:
    vpath = find_existing(here, VEHICLES_FILE_CANDIDATES)
    if not vpath:
        return None
    try:
        v = pd.read_csv(vpath)
        if {"vendor","vehicle_id"}.issubset(set(v.columns)):
            return v
    except Exception:
        return None
    return None

# ---------------- generation ----------------

def negbin(mean: float, k: float, rng: np.random.Generator) -> int:
    if mean <= 0: return 0
    lam = rng.gamma(shape=k, scale=mean/max(k,1e-6))
    return int(rng.poisson(lam))


def pick_vehicle_id(vehicles: Optional[pd.DataFrame], vendor: str, rng: random.Random) -> Optional[str]:
    if vehicles is None: return None
    vsub = vehicles[vehicles["vendor"] == vendor]
    if vsub.empty: return None
    if "odometer_km" in vsub.columns:
        w = pd.to_numeric(vsub["odometer_km"], errors="coerce").fillna(1.0).astype(float)
        probs = w / w.sum()
        idx = np.random.default_rng(rng.randrange(1,10**9)).choice(vsub.index.values, p=probs.values)
        return str(vsub.loc[idx, "vehicle_id"])
    return str(vsub.sample(1, random_state=rng.randrange(1,10**9))["vehicle_id"].iloc[0])


def generate_status_changes() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rr = random.Random(SEED)
    here = Path(__file__).resolve().parent

    hourly, svc = load_hourly_and_meta(here)
    vehicles = load_vehicles(here)

    records: List[Dict[str,object]] = []
    eid = 1

    area_type_map = svc.set_index("service_area_id")["area_type"].to_dict()

    # Active vehicle estimate per vendor-area (from vehicles.csv if available)
    active_by_vendor_area: Dict[Tuple[str,int], int] = {}
    if vehicles is not None and {"service_area_id_home","vendor"}.issubset(set(vehicles.columns)):
        tmp = vehicles.groupby(["vendor","service_area_id_home"]).size()
        for (v, aid), n in tmp.items():
            active_by_vendor_area[(v, int(aid))] = int(n)

    # precompute day-level noise
    for row in hourly.itertuples(index=False):
        d: date = row.day
        break
    days = sorted(hourly["day"].unique())
    day_noise = {d: rng.gamma(shape=6.0, scale=1.0/6.0) for d in days}

    for row in hourly.itertuples(index=False):
        d: date = row.day
        h: int = int(row.hour)
        vendor: str = str(row.vendor)
        trips: float = float(row.trips)
        aid: int = int(row.service_area_id)
        can: str = str(row.community_area_name)
        atype: str = str(row.area_type).lower() if hasattr(row, 'area_type') else 'residential'

        band = hour_band(h)
        month = d.month

        # Estimate active vehicles in this area for vendor
        act_key = (vendor, aid)
        active = active_by_vendor_area.get(act_key, max(3, int(trips / 20)))  # heuristic if no vehicles.csv

        # Expected counts per type
        # Types driven by trips
        lam_batt = (trips * (PER_1K_TRIPS["battery_low"] / 1000.0)
                    * HOUR_BAND_MULT["battery_low"][band] * AREA_MULT["battery_low"].get(atype,1.0)
                    * VENDOR_MULT.get(vendor,1.0) * day_noise[d])
        lam_rebal = (trips * (PER_1K_TRIPS["rebalance"] / 1000.0)
                     * HOUR_BAND_MULT["rebalance"][band] * AREA_MULT["rebalance"].get(atype,1.0)
                     * VENDOR_MULT.get(vendor,1.0) * day_noise[d])
        # Types driven by active fleet
        lam_off = (active * PER_VEHICLE_PER_HOUR["offline"]
                   * HOUR_BAND_MULT["offline"][band] * AREA_MULT["offline"].get(atype,1.0)
                   * VENDOR_MULT.get(vendor,1.0) * day_noise[d])
        lam_maint = (active * PER_VEHICLE_PER_HOUR["maintenance"]
                     * HOUR_BAND_MULT["maintenance"][band] * AREA_MULT["maintenance"].get(atype,1.0)
                     * VENDOR_MULT.get(vendor,1.0) * day_noise[d])
        # Deploy/Retrieve per active
        lam_dep = 0.0; lam_ret = 0.0
        if 6 <= h <= 10:
            lam_dep = active * DEPLOY_PER_ACTIVE * (1.0 if band=="morning" else 0.8)
        if h >= 21 or h <= 1:
            lam_ret = active * RETRIEVE_PER_ACTIVE * (1.0 if band in ("night","late_night") else 0.7)

        # Hourly noise
        hour_noise = rng.lognormal(0.0, 0.12)
        lam_batt *= hour_noise; lam_rebal *= hour_noise; lam_off *= hour_noise; lam_maint *= hour_noise
        lam_dep *= hour_noise; lam_ret *= hour_noise

        # Draw counts (NegBin)
        nbk = NB_K
        n_batt = negbin(lam_batt, nbk, rng)
        n_rebal = negbin(lam_rebal, nbk, rng)
        n_off   = negbin(lam_off,   nbk, rng)
        n_maint = negbin(lam_maint, nbk, rng)
        n_dep   = negbin(lam_dep,   nbk, rng)
        n_ret   = negbin(lam_ret,   nbk, rng)

        # Helper for record append
        def add_record(event_type: str,
                       occurred: datetime,
                       subtype: Optional[str]=None,
                       handle_minutes: Optional[int]=None,
                       battery_before: Optional[int]=None,
                       battery_after: Optional[int]=None,
                       move_km: Optional[float]=None,
                       offline_min: Optional[int]=None,
                       compliance_flag: Optional[bool]=None,
                       cost_usd: Optional[float]=None):
            nonlocal eid
            # SLA
            sla = SLA_MINUTES.get(event_type)
            sla_met_int = None
            handled_at_str = ""
            if handle_minutes is not None:
                sla_met_int = 1 if (sla is not None and handle_minutes <= sla) else 0 if sla is not None else None
                handled_at = occurred + timedelta(minutes=int(handle_minutes))
                handled_at_str = handled_at.strftime("%Y-%m-%d %H:%M:%S")

            vehicle_id = pick_vehicle_id(vehicles, vendor, rr)
            ops_team = rr.choice(OPS_TEAMS)

            records.append({
                "event_id": eid,
                "vendor": vendor,
                "vehicle_id": vehicle_id,
                "event_type": event_type,
                "event_subtype": subtype or "",
                "occurred_at": occurred.strftime("%Y-%m-%d %H:%M:%S"),
                "handled_at": handled_at_str,
                "handle_minutes": int(handle_minutes) if handle_minutes is not None else None,
                "sla_minutes": sla,
                "sla_met_int": sla_met_int,
                "service_area_id": int(aid),
                "community_area_name": can,
                "area_type": atype,
                "hour_band": band,
                "weekday": int(datetime(d.year, d.month, d.day).weekday()),
                "move_km_est": round(float(move_km),2) if move_km is not None else None,
                "battery_pct_before": battery_before,
                "battery_pct_after": battery_after,
                "offline_minutes": int(offline_min) if offline_min is not None else None,
                "ops_team": ops_team,
                "compliance_flag": bool(compliance_flag) if compliance_flag is not None else None,
                "cost_usd": round(float(cost_usd),2) if cost_usd is not None else None,
                "workload_trips_hour": float(trips),
                "day": d,
                "hour": h,
            })
            eid += 1

        # Draw events and fill details
        # battery_low
        for _ in range(n_batt):
            minute = rr.randint(0,59); second = rr.randint(0,59)
            occurred = datetime(d.year,d.month,d.day,h,minute,second)
            mu,sigma = HANDLE_LOGN["battery_low"]
            adj = 1.0
            if atype=="downtown": adj *= 1.1
            if band in ("evening","night"): adj *= 1.15
            handle = max(10, int(np.random.lognormal(mu,sigma) * adj))
            b_before = rr.randint(5,25); b_after = rr.randint(55,95)
            cost = 7.5 + rr.random()*8.0
            add_record("battery_low", occurred, None, handle, b_before, b_after, None, None, False, cost)

        # rebalance
        for _ in range(n_rebal):
            minute = rr.randint(0,59); second = rr.randint(0,59)
            occurred = datetime(d.year,d.month,d.day,h,minute,second)
            mu,sigma = HANDLE_LOGN["rebalance"]
            adj = 1.0
            if atype=="downtown": adj *= 1.15
            if band=="evening": adj *= 1.1
            handle = max(15, int(np.random.lognormal(mu,sigma) * adj))
            move = max(0.1, float(np.random.lognormal(math.log(2.0), 0.5)))
            subtype = rr.choice(REB_SUBTYPES)
            compliance = (subtype=="clear_footpath")
            cost = 6.0 + move*1.4
            add_record("rebalance", occurred, subtype, handle, None, None, move, None, compliance, cost)

        # offline
        for _ in range(n_off):
            minute = rr.randint(0,59); second = rr.randint(0,59)
            occurred = datetime(d.year,d.month,d.day,h,minute,second)
            mu,sigma = HANDLE_LOGN["offline"]
            adj = 1.0
            if band in ("night","late_night"): adj *= 1.25
            handle = max(5, int(np.random.lognormal(mu,sigma) * adj))
            add_record("offline", occurred, None, handle, None, None, None, handle, False, 2.0 + rr.random()*4)

        # maintenance
        for _ in range(n_maint):
            minute = rr.randint(0,59); second = rr.randint(0,59)
            occurred = datetime(d.year,d.month,d.day,h,minute,second)
            mu,sigma = HANDLE_LOGN["maintenance"]
            handle = max(60, int(np.random.lognormal(mu,sigma)))
            add_record("maintenance", occurred, None, handle, None, None, None, handle, False, 25 + rr.random()*60)

        # deployment (informational)
        for _ in range(n_dep):
            minute = rr.randint(0,59); second = rr.randint(0,59)
            occurred = datetime(d.year,d.month,d.day,h,minute,second)
            add_record("deployment", occurred, "morning_push", None, rr.randint(80,100), rr.randint(80,100), None, None, False, 0.0)

        # retrieval (informational)
        for _ in range(n_ret):
            minute = rr.randint(0,59); second = rr.randint(0,59)
            occurred = datetime(d.year,d.month,d.day,h,minute,second)
            add_record("retrieval", occurred, "sweep", None, rr.randint(10,40), rr.randint(10,40), None, None, True, 0.0)

    df = pd.DataFrame.from_records(records)
    return df

# ---------------- main ----------------

def main():
    here = Path(__file__).resolve().parent
    df = generate_status_changes()
    out_path = here / "status_changes.csv"
    # enforce column order
    cols = [
        "event_id","vendor","vehicle_id","event_type","event_subtype",
        "occurred_at","handled_at","handle_minutes","sla_minutes","sla_met_int",
        "service_area_id","community_area_name","area_type","hour_band","weekday",
        "move_km_est","battery_pct_before","battery_pct_after",
        "offline_minutes","ops_team","compliance_flag","cost_usd",
        "workload_trips_hour","day","hour"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d status changes to %s", len(df), out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        raise
