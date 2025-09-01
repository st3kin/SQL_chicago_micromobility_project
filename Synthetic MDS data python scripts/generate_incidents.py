#!/usr/bin/env python3
"""
Zero-setup generator for a realistic, analysis-rich `incidents.csv`.

• No CLI flags. Place this file next to your trips CSV (and optionally
  `hourly_demand_by_vendor.csv`, `vehicles.csv`, `service_areas.csv`) and run:

    python3 generate_incidents_nointput.py

• Calibrates to your *actual* demand: prefers `hourly_demand_by_vendor.csv` if present;
  otherwise derives hourly demand from the trips CSV.
• Produces nuanced incident patterns:
    - Types: battery_failure, crash, illegal_parking, vandalism, theft, complaint
    - Rates per 1k trips vary by hour band, area_type, month (seasonality), vendor
    - Over-dispersion via Negative Binomial (not flat), plus day/hour noise
    - Resolution times depend on incident type, vendor, area_type & workload
• If `vehicles.csv` is present, incidents pick vehicle_ids from the same vendor,
  weighted by rough usage (odometer_km) for realism.

Output columns
--------------
incident_id, vendor, vehicle_id, occurred_at, incident_type, severity,
service_area_id, community_area_name, area_type, report_source,
resolved_at, resolution, cost_usd, trip_related

Also easy-to-join helpers: day (date), hour (0-23)

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

# ---------------- config (edit in file if desired) ----------------
SEED = 44
NB_K = 7.0   # negbin dispersion (smaller => more variance)

# Baseline incident rates per 1,000 trips (before multipliers)
BASE_PER_1K = {
    "battery_failure": 0.35,
    "crash":           0.22,
    "illegal_parking": 0.45,
    "vandalism":       0.18,
    "theft":           0.06,
    "complaint":       0.20,
}

# Hour-band multipliers by type
HOUR_BAND_MULT = {
    "battery_failure": {"late_night":0.9,  "morning":1.0,  "midday":1.0,  "evening":1.05, "night":1.1},
    "crash":           {"late_night":0.8,  "morning":1.2,  "midday":1.05, "evening":1.35, "night":1.1},
    "illegal_parking": {"late_night":1.25, "morning":0.9,  "midday":1.0,  "evening":1.2,  "night":1.35},
    "vandalism":       {"late_night":1.5,  "morning":0.7,  "midday":0.8,  "evening":1.1,  "night":1.3},
    "theft":           {"late_night":1.4,  "morning":0.7,  "midday":0.8,  "evening":1.1,  "night":1.3},
    "complaint":       {"late_night":1.1,  "morning":1.0,  "midday":1.0,  "evening":1.1,  "night":1.2},
}

# Area-type multipliers by type
AREA_MULT = {
    "battery_failure": {"downtown":1.05, "campus":1.0,  "industrial":0.95, "residential":1.0},
    "crash":           {"downtown":1.25, "campus":1.05, "industrial":0.95, "residential":0.95},
    "illegal_parking": {"downtown":1.35, "campus":1.25, "industrial":0.8,  "residential":1.0},
    "vandalism":       {"downtown":1.1,  "campus":1.0,  "industrial":1.2,  "residential":1.1},
    "theft":           {"downtown":1.2,  "campus":1.1,  "industrial":1.2,  "residential":1.0},
    "complaint":       {"downtown":1.2,  "campus":1.1,  "industrial":0.9,  "residential":1.0},
}

# Month (seasonality) multipliers by type (winter fewer trips but more battery issues)
MONTH_MULT = {
    "battery_failure": {1:1.25,2:1.25,3:1.15,4:1.05,5:1.0,6:0.95,7:0.9,8:0.9,9:0.95,10:1.0,11:1.1,12:1.2},
    "crash":           {1:0.8, 2:0.85,3:0.9, 4:1.0, 5:1.05,6:1.1, 7:1.15,8:1.15,9:1.05,10:1.0,11:0.9,12:0.85},
    "illegal_parking": {1:0.9, 2:0.9, 3:0.95,4:1.0, 5:1.05,6:1.1, 7:1.2, 8:1.2, 9:1.1, 10:1.0,11:0.95,12:0.9},
    "vandalism":       {1:1.0, 2:1.0, 3:1.0, 4:1.05,5:1.05,6:1.1, 7:1.15,8:1.2, 9:1.1, 10:1.05,11:1.0,12:1.0},
    "theft":           {1:1.0, 2:1.0, 3:1.05,4:1.1, 5:1.15,6:1.2, 7:1.25,8:1.25,9:1.15,10:1.05,11:1.0,12:1.0},
    "complaint":       {1:0.95,2:0.95,3:1.0, 4:1.0, 5:1.05,6:1.05,7:1.1, 8:1.1, 9:1.05,10:1.0,11:1.0,12:0.95},
}

# Vendor idiosyncrasies (tune small deltas only)
VENDOR_MULT = {"Lyft":1.0, "Lime":0.98, "Spin":1.05, "Coco":1.08}

# Severity distribution per type (1=low, 2=med, 3=high)
SEVERITY_P = {
    "battery_failure": [0.60, 0.30, 0.10],
    "crash":           [0.40, 0.45, 0.15],
    "illegal_parking": [0.70, 0.25, 0.05],
    "vandalism":       [0.55, 0.35, 0.10],
    "theft":           [0.30, 0.50, 0.20],
    "complaint":       [0.75, 0.22, 0.03],
}

# Resolution time distributions (minutes) ~ lognormal parameters per type
# mean ~ exp(mu), with sigma controlling tail; later adjusted by vendor/area/hour load
RESOLVE_LOGN = {
    "battery_failure": (math.log(45), 0.5),
    "crash":           (math.log(90), 0.6),
    "illegal_parking": (math.log(75), 0.55),
    "vandalism":       (math.log(240),0.7),
    "theft":           (math.log(480),0.8),
    "complaint":       (math.log(60), 0.5),
}

# Resolution text options per type
RESOLUTION_TEXT = {
    "battery_failure": ["Reset/BMS","Swap battery","On-site fix","Workshop"],
    "crash":           ["Rider assisted","EMS notified","Vehicle recovered","No action"],
    "illegal_parking": ["Reparked","Picked up","No issue found","Ticketed"],
    "vandalism":       ["Cleaned","Parts replaced","Workshop","Police report"],
    "theft":           ["Recovered","Insurance claim","Police report"],
    "complaint":       ["Contacted rider","Contacted reporter","No issue found"],
}

REPORT_SOURCES = ["citizen_report","rider_report","field_op","police"]

# ---------------- file discovery ----------------
TRIPS_FILE_CANDIDATES = ["e_scooter_trips.csv","e-scooter-trips.csv","chicago_trips.csv","trips.csv"]
HOURLY_FILE_CANDIDATES = ["hourly_demand_by_vendor.csv","mds_export/hourly_demand_by_vendor.csv"]
SERVICE_AREAS_FILE_CANDIDATES = ["service_areas.csv","mds_export/service_areas.csv","../mds_export/service_areas.csv"]
VEHICLES_FILE_CANDIDATES = ["vehicles.csv","mds_export/vehicles.csv","../mds_export/vehicles.csv"]

LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("incidents_auto")

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
            if set(["vendor"]).issubset(set([c.lower() for c in head.columns.astype(str)])):
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


def load_trips_for_hourly(trips_path: Path) -> pd.DataFrame:
    """Return hourly demand if hourly file not present. Columns: day, hour, service_area_id, community_area_name, vendor, trips."""
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
        # normalize column types
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
        # infer neutral area_type if missing
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
        if "vendor" in v.columns and "vehicle_id" in v.columns:
            return v
    except Exception:
        return None
    return None

# ---------------- generation ----------------

def negbin(mean: float, k: float, rng: np.random.Generator) -> int:
    if mean <= 0: return 0
    lam = rng.gamma(shape=k, scale=mean/max(k,1e-6))
    return int(rng.poisson(lam))


def type_probability_vector(vendor: str, area_type: str, h: int, month: int) -> Dict[str,float]:
    # construct relative probabilities across types to softly vary mix
    base = BASE_PER_1K.copy()
    for t in base:
        base[t] *= HOUR_BAND_MULT[t][hour_band(h)]
        base[t] *= AREA_MULT[t].get(area_type, 1.0)
        base[t] *= MONTH_MULT[t].get(month, 1.0)
        base[t] *= VENDOR_MULT.get(vendor, 1.0)
    s = sum(base.values()) or 1.0
    return {k: v/s for k,v in base.items()}


def pick_vehicle_id(vehicles: Optional[pd.DataFrame], vendor: str, rng: random.Random) -> Optional[str]:
    if vehicles is None: return None
    vsub = vehicles[vehicles["vendor"] == vendor]
    if vsub.empty: return None
    # weight by odometer if available
    if "odometer_km" in vsub.columns:
        w = pd.to_numeric(vsub["odometer_km"], errors="coerce").fillna(1.0).astype(float)
        probs = w / w.sum()
        idx = np.random.default_rng(rng.randrange(1,10**9)).choice(vsub.index.values, p=probs.values)
        return str(vsub.loc[idx, "vehicle_id"])
    return str(vsub.sample(1, random_state=rng.randrange(1,10**9))["vehicle_id"].iloc[0])


def generate_incidents() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rr = random.Random(SEED)
    here = Path(__file__).resolve().parent

    hourly, svc = load_hourly_and_meta(here)
    vehicles = load_vehicles(here)

    if vehicles is not None:
        logging.info("Using vehicles.csv for vehicle_id attribution")

    records: List[Dict[str,object]] = []
    inc_id = 1

    # Precompute area_type map
    at_map = svc.set_index("service_area_id")["area_type"].to_dict()

    # Daily noise to avoid flatness day-to-day
    # Build a dict of day -> gamma noise factor
    days = sorted(hourly["day"].unique())
    day_noise_map = {d: rng.gamma(shape=6.0, scale=1.0/6.0) for d in days}

    # Iterate hourly demand
    for row in hourly.itertuples(index=False):
        d: date = row.day
        h: int = int(row.hour)
        vendor: str = str(row.vendor)
        trips: float = float(row.trips)
        aid: int = int(row.service_area_id)
        can: str = str(row.community_area_name)
        atype: str = str(row.area_type).lower() if hasattr(row, 'area_type') else 'residential'

        # expected count scaling
        month = d.month
        # Build per-type expected rate
        type_mix = type_probability_vector(vendor, atype, h, month)
        # We modulate overall intensity by trips and a global base rate (sum of BASE_PER_1K)
        base_intensity = sum(BASE_PER_1K.values()) / 1000.0  # per trip
        mean_total = trips * base_intensity
        mean_total *= day_noise_map[d]
        # hour-level noise
        mean_total *= rng.lognormal(0.0, 0.12)

        total_incidents = negbin(mean_total, NB_K, rng)
        if total_incidents <= 0:
            continue

        # Split into types according to type_mix via multinomial
        probs = np.array([type_mix[t] for t in BASE_PER_1K.keys()], dtype=float)
        probs = probs / probs.sum()
        type_counts = np.random.multinomial(total_incidents, probs)
        t_order = list(BASE_PER_1K.keys())

        for t, n in zip(t_order, type_counts):
            if n <= 0: continue
            # Severity distribution
            sev_p = np.array(SEVERITY_P[t], dtype=float)
            sev_vals = np.array([1,2,3])
            # Resolution time parameters
            mu, sigma = RESOLVE_LOGN[t]
            # Adjust resolution by area_type and hour load (simple multipliers)
            res_adj = 1.0
            if atype == "downtown": res_adj *= 1.15
            if hour_band(h) in ("evening","night"): res_adj *= 1.10

            for _ in range(int(n)):
                # time within the hour
                minute = rr.randint(0,59); second = rr.randint(0,59)
                occurred = datetime(d.year, d.month, d.day, h, minute, second)
                # severity
                severity = int(np.random.choice(sev_vals, p=sev_p))
                # resolution time
                res_minutes = max(5, int(np.random.lognormal(mean=mu, sigma=sigma) * res_adj))
                resolved = occurred + timedelta(minutes=res_minutes)
                # resolution text & source
                resolution = rr.choice(RESOLUTION_TEXT[t])
                source = rr.choices(REPORT_SOURCES, weights=[0.45,0.25,0.25,0.05], k=1)[0]
                # cost model (optional)
                cost = 0.0
                if t == "illegal_parking":
                    cost = float(10 + rr.random()*20)  # towing/ops cost proxy
                elif t == "vandalism":
                    cost = float(30 + rr.random()*120)
                elif t == "theft":
                    cost = float(150 + rr.random()*400)
                elif t == "crash":
                    cost = float(20 + rr.random()*200)
                elif t == "battery_failure":
                    cost = float(10 + rr.random()*60)

                vehicle_id = pick_vehicle_id(vehicles, vendor, rr)
                trip_related = (t in ("crash","complaint"))

                records.append({
                    "incident_id": inc_id,
                    "vendor": vendor,
                    "vehicle_id": vehicle_id,
                    "occurred_at": occurred.strftime("%Y-%m-%d %H:%M:%S"),
                    "incident_type": t,
                    "severity": severity,
                    "service_area_id": aid,
                    "community_area_name": can,
                    "area_type": atype,
                    "report_source": source,
                    "resolved_at": resolved.strftime("%Y-%m-%d %H:%M:%S"),
                    "resolution": resolution,
                    "cost_usd": round(cost, 2),
                    "trip_related": bool(trip_related),
                    "day": d,
                    "hour": h,
                })
                inc_id += 1

    df = pd.DataFrame.from_records(records)
    return df


def main():
    here = Path(__file__).resolve().parent
    df = generate_incidents()
    out_path = here / "incidents.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d incidents to %s", len(df), out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        raise
