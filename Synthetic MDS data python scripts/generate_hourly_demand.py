#!/usr/bin/env python3
"""
Zero-setup generator for a realistic, non-flat `hourly_demand_by_vendor` table.

• No CLI args. Drop this script into the folder with your trips CSV and (optionally)
  `service_areas.csv`, then run:  python3 generate_hourly_demand_nointput.py
• Auto-detects your trips file & columns (works with Chicago e-scooter schema).
• Calibrates to your data (vendor mix by month, hour-of-day, area weights),
  with seasonality, weekend boost, vendor×hour and vendor×area interactions,
  over-dispersion (negative binomial), and light noise on areas/hours.
• Outputs next to the trips CSV: `hourly_demand_by_vendor.csv`

Join keys: day (date), hour (0–23), service_area_id, community_area_name, vendor, trips

Editables below (SEED, CITY_SCALE, NB_K) if you want to tune variance/volume later.

Dependencies: Python 3.8+, pandas, numpy (pyarrow optional)
"""

import logging
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------- config you can tweak in-code (no CLI needed) ----------------
SEED = 42           # RNG seed (deterministic output)
CITY_SCALE = 1.0    # multiply city totals (1.0 = calibrated to observed median day)
NB_K = 8.0          # negative-binomial dispersion (smaller => spikier)

# If your files use different names, add them to these patterns:
TRIPS_FILE_CANDIDATES = [
    "e_scooter_trips.csv",
    "e-scooter-trips.csv",
    "chicago_trips.csv",
    "trips.csv",
]
SERVICE_AREAS_FILE_CANDIDATES = [
    "service_areas.csv",
    "mds_export/service_areas.csv",
]

# ---------------- constants ----------------
LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("hourly_demand_auto")

VENDOR_MAP = {
    "lyft": "Lyft", "lime": "Lime", "spin": "Spin", "coco": "Coco",
    # common alternates mapped for realism
    "bird": "Coco", "tier": "Coco", "superpedestrian": "Spin"
}

DEFAULT_MONTH_MULT = {1:0.60,2:0.65,3:0.72,4:0.85,5:0.95,6:1.05,7:1.10,8:1.12,9:1.05,10:1.00,11:0.75,12:0.55}
WEEKEND_BOOST = 1.08
MIN_VENDOR_FLOORS = {"Lyft":0.28, "Lime":0.25, "Spin":0.10, "Coco":0.07}

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
    # fallback: first CSV with detectable columns
    for p in path.glob("*.csv"):
        try:
            head = pd.read_csv(p, nrows=0)
            m = detect_columns(head)
            if m["start_time"] and m["vendor"]:
                return p
        except Exception:
            continue
    return None


def detect_columns(df_head: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df_head.columns)
    ncols = {c: norm(c) for c in cols}
    def match(cands: List[str]) -> Optional[str]:
        cands_n = [norm(c) for c in cands]
        for c, nc in ncols.items():
            if nc in cands_n:
                return c
        for c, nc in ncols.items():
            if any(nc.find(x) >= 0 for x in cands_n):
                return c
        return None
    return {
        "start_time": match(["start_time","started_at","trip_start","startdatetime","starttimestamp","startdate","starttime"]),
        "vendor": match(["vendor","provider","provider_name","company","operator"]),
        "start_area_name": match(["start_community_area_name","start_area_name","start_neighborhood","start_zone_name","start_community_area"]),
        "end_area_name": match(["end_community_area_name","end_area_name","end_neighborhood","end_zone_name","end_community_area"]),
        "start_area_num": match(["start_community_area_number","start_area_id","start_area_code"]),
        "end_area_num": match(["end_community_area_number","end_area_id","end_area_code"]),
    }


from typing import Optional, Tuple, Dict
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np

def load_trips(trips_path: Path,
               start: Optional[date] = None,
               end: Optional[date] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load trips and derive calibration fields. Robust to missing start/end area names."""
    try:
        head = pd.read_csv(trips_path, nrows=0)
    except Exception as e:
        raise SystemExit(f"Failed to read {trips_path}: {e}")

    mapping = detect_columns(head)
    missing = [k for k in ("start_time","vendor") if mapping[k] is None]
    if missing:
        raise SystemExit(f"Could not detect required columns: {missing}. Found={list(head.columns)}")

    usecols = [c for c in (mapping["start_time"], mapping["vendor"],
                           mapping["start_area_name"], mapping["end_area_name"],
                           mapping["start_area_num"], mapping["end_area_num"]) if c]

    try:
        df = pd.read_csv(trips_path, usecols=usecols, engine="pyarrow")
    except Exception:
        df = pd.read_csv(trips_path, usecols=usecols)

    # Parse start time (no deprecated infer flag)
    df[mapping["start_time"]] = pd.to_datetime(df[mapping["start_time"]], errors="coerce")
    df = df.dropna(subset=[mapping["start_time"]])

    df["__vendor__"] = df[mapping["vendor"]].map(canon_vendor)
    df["__day__"] = df[mapping["start_time"]].dt.date
    df["__hour__"] = df[mapping["start_time"]].dt.hour.astype(int)

    sname = mapping["start_area_name"]; ename = mapping["end_area_name"]
    snum  = mapping["start_area_num"];  enum  = mapping["end_area_num"]

    if sname or ename:
        # Start with empty strings; prefer start area name; fallback to end area name
        area = pd.Series([""] * len(df), index=df.index, dtype="object")
        if sname is not None:
            area = df[sname].astype(str).str.strip()
        if ename is not None:
            area = area.where(area.notna() & (area.astype(str).str.len() > 0),
                              df[ename].astype(str).str.strip())
        df["__area_name__"] = area.fillna("").astype(str).str.strip()
    else:
        # Use numeric areas if names aren’t present
        if snum is not None:
            num_series = pd.to_numeric(df[snum], errors="coerce")
        elif enum is not None:
            num_series = pd.to_numeric(df[enum], errors="coerce")
        else:
            num_series = pd.Series(np.nan, index=df.index)
        df["__area_num__"] = num_series

    # Optional date window
    if start is not None:
        df = df[df["__day__"] >= start]
    if end is not None:
        df = df[df["__day__"] <= end]

    logger.info("Trips rows used for calibration: %d", len(df))
    return df, mapping



def load_service_areas(base_dir: Path) -> Optional[pd.DataFrame]:
    for cand in SERVICE_AREAS_FILE_CANDIDATES:
        p = (base_dir / cand).resolve()
        if p.exists():
            try:
                df = pd.read_csv(p)
                if "service_area_id" in df.columns and "community_area_name" in df.columns:
                    if "area_type" not in df.columns:
                        df["area_type"] = "residential"
                    logger.info("Using service_areas from %s", p)
                    return df
            except Exception:
                pass
    logger.warning("service_areas.csv not found; inferring areas from trips")
    return None


def build_area_catalog(trips: pd.DataFrame, svc: Optional[pd.DataFrame]) -> pd.DataFrame:
    if svc is not None:
        cat = svc[["service_area_id","community_area_name","area_type"]].copy()
        cat["__nname__"] = cat["community_area_name"].astype(str).str.strip().str.lower()
        return cat
    if "__area_name__" in trips.columns:
        names = trips["__area_name__"].dropna().astype(str).str.strip()
        names = names[names != ""].unique().tolist()
        rows = []
        for i, nm in enumerate(sorted(names), 1):
            rows.append({"service_area_id": i, "community_area_name": nm, "area_type": "residential", "__nname__": nm.lower()})
        return pd.DataFrame(rows)
    else:
        nums = trips["__area_num__"].dropna().astype(int).unique().tolist()
        rows = []
        for n in sorted(nums):
            rows.append({"service_area_id": int(n), "community_area_name": f"CA {n:02d}", "area_type": "residential", "__nname__": f"ca {n:02d}"})
        return pd.DataFrame(rows)


def normalize_with_floors(weights: Dict[str, float], floors: Dict[str, float]) -> Dict[str, float]:
    floors = {k: floors.get(k, 0.0) for k in weights}
    total_floor = sum(floors.values())
    s = sum(max(0.0, v) for v in weights.values()) or 1.0
    out = {k: floors[k] + (1 - total_floor) * (max(0.0, weights[k]) / s) for k in weights}
    ss = sum(out.values()) or 1.0
    return {k: out[k] / ss for k in out}


def compute_calibration(trips: pd.DataFrame, areas: pd.DataFrame) -> Dict[str, object]:
    trips = trips.copy()
    trips["__month__"] = trips["__day__"].apply(lambda d: f"{d.year}-{d.month:02d}")
    month_mix: Dict[str, Dict[str, float]] = {}
    for mk, sub in trips.groupby("__month__"):
        s = sub["__vendor__"].value_counts(normalize=True)
        base = {v: float(s.get(v, 0.0)) for v in ("Lyft","Lime","Spin","Coco")}
        month_mix[mk] = normalize_with_floors(base, MIN_VENDOR_FLOORS)

    hour_shape = trips["__hour__"].value_counts(normalize=True).reindex(range(24), fill_value=1/24).to_dict()

    if "__area_name__" in trips.columns:
        t_counts = trips.groupby(trips["__area_name__"].str.strip().str.lower()).size()
        areas = areas.copy(); areas.set_index("__nname__", inplace=True)
        aligned = t_counts.reindex(areas.index, fill_value=0)
        w = (aligned + 0.5); area_w = (w / w.sum()).to_dict()
    else:
        t_counts = trips.groupby("__area_num__").size()
        areas = areas.copy(); areas.set_index("service_area_id", inplace=True)
        aligned = t_counts.reindex(areas.index, fill_value=0)
        w = (aligned + 0.5); area_w = (w / w.sum()).to_dict()

    day_totals = trips.groupby("__day__").size()
    daily_median = float(day_totals.median()) if not day_totals.empty else 50000.0

    month_totals = trips.groupby(trips["__day__"].apply(lambda d: d.month)).size()
    if not month_totals.empty:
        month_mult = {m: (month_totals.get(m, 0) / month_totals.mean()) for m in range(1,13)}
    else:
        month_mult = DEFAULT_MONTH_MULT

    if not day_totals.empty:
        dow = trips.groupby(trips["__day__"].apply(lambda d: d.weekday())).size()
        wk = float(dow.get(5, 1) + dow.get(6, 1)) / 2.0
        wd = float(np.mean([dow.get(i, 1) for i in range(0,5)]))
        weekend_boost = (wk / wd) if wd > 0 else WEEKEND_BOOST
    else:
        weekend_boost = WEEKEND_BOOST

    return dict(month_mix=month_mix, hour_shape=hour_shape, area_w=area_w,
                daily_median=daily_median, month_mult=month_mult, weekend_boost=weekend_boost)


def hour_band(h: int) -> str:
    if h <= 5: return "late_night"
    if h <= 9: return "morning"
    if h <= 15: return "midday"
    if h <= 19: return "evening"
    return "night"


def vendor_hour_bias(v: str, h: int) -> float:
    b = hour_band(h)
    if v == "Lyft": return {"morning":1.05,"midday":1.00,"evening":1.08,"night":0.98,"late_night":0.95}[b]
    if v == "Lime": return {"morning":1.02,"midday":1.04,"evening":1.05,"night":1.00,"late_night":0.96}[b]
    if v == "Spin": return {"morning":0.95,"midday":0.98,"evening":1.05,"night":1.12,"late_night":1.22}[b]
    if v == "Coco": return {"morning":0.92,"midday":0.96,"evening":1.08,"night":1.18,"late_night":1.28}[b]
    return 1.0


def vendor_area_bias(v: str, atype: str) -> float:
    at = (atype or "residential").lower(); m = 1.0
    if v == "Spin" and at in ("residential","industrial"): m *= 1.10
    if v == "Coco" and at == "campus": m *= 1.25
    if v == "Lime" and at == "downtown": m *= 1.08
    return m


def neg_binomial(mean: float, k: float, rng: np.random.Generator) -> int:
    if mean <= 0: return 0
    shape = k
    scale = mean / k
    lam = rng.gamma(shape, scale)
    return int(rng.poisson(lam))

# ---------------- main ----------------

def main():
    rng = np.random.default_rng(SEED)

    here = Path(__file__).resolve().parent
    trips_path = find_existing(here, TRIPS_FILE_CANDIDATES)
    if not trips_path:
        raise SystemExit(
            "Could not find a trips CSV. Put this script next to your trips file (e.g., 'e_scooter_trips.csv')."
        )
    logger.info("Found trips CSV: %s", trips_path)

    trips, mapping = load_trips(trips_path)

    svc = load_service_areas(trips_path.parent)
    areas = build_area_catalog(trips, svc)
    area_type_map = areas.set_index("service_area_id")["area_type"].to_dict()

    cal = compute_calibration(trips, areas)
    logger.info("Calibration: daily_median≈%.0f, weekend_boost≈%.2f", cal["daily_median"], cal["weekend_boost"])

    start = trips["__day__"].min()
    end = trips["__day__"].max()
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)

    hour_weights = np.array([cal["hour_shape"].get(h, 1/24) for h in range(24)], dtype=float)
    hour_weights = hour_weights / hour_weights.sum()

    area_ids = areas["service_area_id"].tolist()
    area_weights = np.array([
        cal["area_w"].get(
            aid if "__area_name__" not in trips.columns else areas.loc[areas["service_area_id"]==aid, "__nname__"].iloc[0],
            1.0
        ) for aid in area_ids
    ], dtype=float)
    area_weights = area_weights / area_weights.sum()

    records: List[Dict[str, object]] = []
    area_types = [area_type_map.get(aid, "residential") for aid in area_ids]

    for d in days:
        month = d.month
        base = cal["daily_median"]
        m_mult = cal["month_mult"].get(month, DEFAULT_MONTH_MULT.get(month, 1.0))
        dow_mult = cal["weekend_boost"] if datetime(d.year, d.month, d.day).weekday() >= 5 else 1.0
        day_noise = rng.gamma(shape=6.0, scale=1.0/6.0)
        city_total = max(5000, int(base * m_mult * dow_mult * day_noise * CITY_SCALE))

        mk = f"{d.year}-{d.month:02d}"
        vshare = cal["month_mix"].get(mk, MIN_VENDOR_FLOORS)

        for h in range(24):
            hw = hour_weights[h]
            hour_total = city_total * hw
            hour_total *= rng.lognormal(mean=0.0, sigma=0.15)

            aw_noise = rng.lognormal(0.0, 0.12, size=len(area_weights))
            aw = area_weights * aw_noise
            aw = aw / aw.sum()

            for aid, atype, awi in zip(area_ids, area_types, aw):
                area_hour_total = hour_total * awi
                raw = {v: vshare.get(v, 0.0) * vendor_hour_bias(v, h) * vendor_area_bias(v, atype)
                       for v in ("Lyft","Lime","Spin","Coco")}
                shares = normalize_with_floors(raw, MIN_VENDOR_FLOORS)
                for v in ("Lyft","Lime","Spin","Coco"):
                    mean = max(0.0, area_hour_total * shares[v])
                    trips_ct = neg_binomial(mean, NB_K, rng)
                    if trips_ct == 0 and mean > 2 and rng.random() < 0.15:
                        trips_ct = 1
                    if trips_ct > 0:
                        can = areas.loc[areas["service_area_id"] == aid, "community_area_name"].iloc[0]
                        records.append({
                            "day": d,
                            "hour": h,
                            "service_area_id": int(aid),
                            "community_area_name": str(can),
                            "vendor": v,
                            "trips": int(trips_ct),
                        })

    df = pd.DataFrame.from_records(records)
    if df.empty:
        logger.warning("No rows generated; check inputs")
        df = pd.DataFrame(columns=["day","hour","service_area_id","community_area_name","vendor","trips"]) 

    g = df.groupby(["day","hour"], as_index=False)["trips"].sum().rename(columns={"trips":"dayhour_total"})
    df = df.merge(g, on=["day","hour"], how="left")
    df["share_within_day"] = df["trips"] / df["dayhour_total"].replace(0, np.nan)

    g2 = df.groupby(["vendor","day"], as_index=False)["trips"].sum().rename(columns={"trips":"vendor_day_total"})
    df = df.merge(g2, on=["vendor","day"], how="left")
    df["share_within_vendor_day"] = df["trips"] / df["vendor_day_total"].replace(0, np.nan)

    out_path = trips_path.parent / "hourly_demand_by_vendor.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s rows to %s", len(df), out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        raise
