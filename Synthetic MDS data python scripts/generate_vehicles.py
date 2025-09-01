#!/usr/bin/env python3
"""
Zero-setup generator for a realistic, analysis-rich `vehicles.csv` calibrated to your
Chicago e-scooter trips. No CLI flags. Place this file next to your trips CSV and
(optionally) `service_areas.csv`, then run:

    python3 generate_vehicles_nointput.py

It will write `vehicles.csv` in the same folder.

What you get (columns)
----------------------
- vehicle_id                  e.g., LY00001, LI00001, SP00001, CO00001
- vendor                      {Lyft, Lime, Spin, Coco}
- form_factor                 "scooter"
- model                       vendor-specific model name
- battery_wh                  [460, 520, 620] (weighted by vendor)
- battery_swappable           True/False (mostly True)
- service_area_id_home        home area id
- community_area_name_home    home area name
- area_type_home              {downtown, campus, industrial, residential}
- in_service_since            YYYY-MM-DD
- decommissioned_at           YYYY-MM-DD or blank
- status                      {active, maintenance, decommissioned}
- firmware_version            semantic-ish version per vendor
- iot_modem                   {Quectel, u-blox, Sierra}
- odometer_km                 estimated cumulative km based on usage
- est_daily_trips_target      target trips/vehicle/day used for sizing
- est_daily_energy_wh         rough energy per day (from trips & vendor)
- battery_cycles_est          lifetime cycles estimate

How it stays realistic/non-flat
--------------------------------
• Learns vendor share, daily volume and area mix from your `e_scooter_trips.csv`.
• Sizes each vendor's fleet using a trips-per-vehicle/day target (vendor-specific).
• Distributes vehicles to home areas using vendor×area demand share.
• Rich per-vehicle attributes (battery, firmware, aging, odometer) tied to usage.

Dependencies: Python 3.8+, pandas, numpy (pyarrow optional)
"""

from __future__ import annotations
import logging
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd

# ---------------- config (edit in file if needed) ----------------
SEED = 43  # deterministic RNG seed
# Trips per vehicle per day target by vendor (used to size fleet)
TPV_TARGET = {"Lyft": 7.5, "Lime": 7.0, "Spin": 6.5, "Coco": 5.0}
MIN_FLEET_PER_VENDOR = 60  # keep challengers visible
# Battery portfolios by vendor (weights)
BATTERY_PORTFOLIO = {
    "Lyft": [(520, 0.6), (620, 0.3), (460, 0.1)],
    "Lime": [(520, 0.5), (460, 0.3), (620, 0.2)],
    "Spin": [(460, 0.5), (520, 0.4), (620, 0.1)],
    "Coco": [(460, 0.6), (520, 0.3), (620, 0.1)],
}
# Model portfolios by vendor
MODEL_PORTFOLIO = {
    "Lyft": [("Segway Max Gen2",0.55),("Segway Max Gen3",0.35),("Okai ES400",0.10)],
    "Lime": [("Gen4",0.65),("Gen3",0.25),("Okai ES400",0.10)],
    "Spin": [("S-300",0.50),("S-200",0.35),("S-400",0.15)],
    "Coco": [("City X",0.40),("Campus One",0.35),("Street Lite",0.25)],
}
FIRMWARE_MAJOR = {"Lyft": 1, "Lime": 2, "Spin": 1, "Coco": 1}
IOT_MODEMS = ["Quectel","u-blox","Sierra"]
STATUS_WEIGHTS = [("active", 0.90), ("maintenance", 0.08), ("decommissioned", 0.02)]
# Energy per trip (Wh) by vendor, used for battery cycles estimate
ENERGY_PER_TRIP_WH = {"Lyft": 35, "Lime": 34, "Spin": 33, "Coco": 30}

LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("vehicles_auto")

VENDOR_MAP = {"lyft":"Lyft","lime":"Lime","spin":"Spin","coco":"Coco","bird":"Coco","tier":"Coco","superpedestrian":"Spin"}

TRIPS_FILE_CANDIDATES = [
    "e_scooter_trips.csv","e-scooter-trips.csv","chicago_trips.csv","trips.csv"
]
SERVICE_AREAS_FILE_CANDIDATES = ["service_areas.csv","../mds_export/service_areas.csv","mds_export/service_areas.csv"]

# ---------------- helpers ----------------

def norm(s: str) -> str:
    return "" if s is None else "".join(ch for ch in str(s).lower() if ch.isalnum())


def canon_vendor(x: object) -> str:
    key = norm("" if pd.isna(x) else x)
    return VENDOR_MAP.get(key, "Coco")


def detect_columns(df_head: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df_head.columns)
    ncols = {c: norm(c) for c in cols}
    def match(cands: List[str]) -> Optional[str]:
        cands_n = [norm(c) for c in cands]
        for c, nc in ncols.items():
            if nc in cands_n: return c
        for c, nc in ncols.items():
            if any(nc.find(x) >= 0 for x in cands_n): return c
        return None
    return {
        "start_time": match(["start_time","started_at","trip_start","startdatetime","starttimestamp","startdate","starttime"]),
        "vendor": match(["vendor","provider","provider_name","company","operator"]),
        "start_area_name": match(["start_community_area_name","start_area_name","start_neighborhood","start_zone_name","start_community_area"]),
        "end_area_name": match(["end_community_area_name","end_area_name","end_neighborhood","end_zone_name","end_community_area"]),
        "start_area_num": match(["start_community_area_number","start_area_id","start_area_code"]),
        "end_area_num": match(["end_community_area_number","end_area_id","end_area_code"]),
        # optional distance columns
        "distance_m": match(["distance_m","trip_distance_m","distance","distance_meters"]),
        "distance_km": match(["distance_km","trip_distance_km","km"]),
    }


def find_existing(path: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = (path / name).resolve()
        if p.exists():
            return p
    for p in path.glob("*.csv"):
        try:
            head = pd.read_csv(p, nrows=0)
            m = detect_columns(head)
            if m["start_time"] and m["vendor"]:
                return p
        except Exception:
            continue
    return None


def load_trips(trips_path: Path) -> Tuple[pd.DataFrame, Dict[str,str]]:
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
                           mapping["start_area_num"], mapping["end_area_num"],
                           mapping["distance_m"], mapping["distance_km"]) if c]
    try:
        df = pd.read_csv(trips_path, usecols=usecols, engine="pyarrow")
    except Exception:
        df = pd.read_csv(trips_path, usecols=usecols)

    df[mapping["start_time"]] = pd.to_datetime(df[mapping["start_time"]], errors="coerce")
    df = df.dropna(subset=[mapping["start_time"]])

    df["__vendor__"] = df[mapping["vendor"]].map(canon_vendor)
    df["__day__"] = df[mapping["start_time"]].dt.date

    # area name preference
    sname = mapping["start_area_name"]; ename = mapping["end_area_name"]
    snum  = mapping["start_area_num"];  enum  = mapping["end_area_num"]
    if sname or ename:
        area = pd.Series([""] * len(df), index=df.index, dtype="object")
        if sname is not None:
            area = df[sname].astype(str).str.strip()
        if ename is not None:
            area = area.where(area.notna() & (area.astype(str).str.len() > 0),
                              df[ename].astype(str).str.strip())
        df["__area_name__"] = area.fillna("").astype(str).str.strip()
    else:
        if snum is not None:
            num_series = pd.to_numeric(df[snum], errors="coerce")
        elif enum is not None:
            num_series = pd.to_numeric(df[enum], errors="coerce")
        else:
            num_series = pd.Series(np.nan, index=df.index)
        df["__area_num__"] = num_series

    # distance (optional)
    avg_km = None
    if mapping["distance_km"]:
        avg_km = pd.to_numeric(df[mapping["distance_km"]], errors="coerce").dropna().mean()
    elif mapping["distance_m"]:
        avg_km = (pd.to_numeric(df[mapping["distance_m"]], errors="coerce").dropna().mean() or 0) / 1000.0
    if avg_km is None or not np.isfinite(avg_km):
        avg_km = 1.9  # reasonable city median

    logger.info("Trips rows for calibration: %d; avg_km≈%.2f", len(df), avg_km)
    return df, mapping, float(avg_km)


def load_service_areas(base_dir: Path) -> Optional[pd.DataFrame]:
    for cand in SERVICE_AREAS_FILE_CANDIDATES:
        p = (base_dir / cand).resolve()
        if p.exists():
            df = pd.read_csv(p)
            if "service_area_id" in df.columns and "community_area_name" in df.columns:
                if "area_type" not in df.columns:
                    df["area_type"] = "residential"
                df["__nname__"] = df["community_area_name"].astype(str).str.strip().str.lower()
                logger.info("Using service_areas from %s", p)
                return df
    logger.warning("service_areas.csv not found; inferring areas from trips")
    return None


def build_area_catalog(trips: pd.DataFrame, svc: Optional[pd.DataFrame]) -> pd.DataFrame:
    if svc is not None:
        return svc[["service_area_id","community_area_name","area_type","__nname__"]].copy()
    if "__area_name__" in trips.columns:
        names = trips["__area_name__"].dropna().astype(str).str.strip()
        names = names[names != ""].unique().tolist()
        rows = []
        for i, nm in enumerate(sorted(names), 1):
            rows.append({"service_area_id": i, "community_area_name": nm, "area_type": "residential", "__nname__": nm.lower()})
        return pd.DataFrame(rows)
    nums = trips["__area_num__"].dropna().astype(int).unique().tolist()
    rows = []
    for n in sorted(nums):
        rows.append({"service_area_id": int(n), "community_area_name": f"CA {n:02d}", "area_type": "residential", "__nname__": f"ca {n:02d}"})
    return pd.DataFrame(rows)

# ---------------- core sizing & generation ----------------

def normalize(weights: Dict[str,float]) -> Dict[str,float]:
    s = sum(max(0.0,v) for v in weights.values()) or 1.0
    return {k: max(0.0, v)/s for k,v in weights.items()}


def choose_weighted(rng: random.Random, items: List[Tuple[object,float]]):
    # items: [(value, weight), ...]
    vals = [v for v,_ in items]
    w = [max(0.0,x) for _,x in items]
    total = sum(w)
    if total <= 0:
        return rng.choice(vals)
    r = rng.random()*total
    acc = 0.0
    for v,wt in zip(vals,w):
        acc += wt
        if r <= acc:
            return v
    return vals[-1]


def vendor_abbrev(v: str) -> str:
    return {"Lyft":"LY","Lime":"LI","Spin":"SP","Coco":"CO"}.get(v, v[:2].upper())


def compute_calibration(trips: pd.DataFrame, areas: pd.DataFrame) -> Dict[str,object]:
    # vendor daily average trips
    vd = trips.groupby(["__vendor__","__day__"]).size().groupby("__vendor__").mean()
    vendor_daily_avg = {v: float(vd.get(v, 0.0)) for v in ("Lyft","Lime","Spin","Coco")}

    # vendor-area mix
    if "__area_name__" in trips.columns:
        t_counts = trips.groupby(["__vendor__", trips["__area_name__"].str.strip().str.lower()]).size()
        areas = areas.copy(); areas.set_index("__nname__", inplace=True)
        mix: Dict[str, Dict[str,float]] = {}
        for v in ("Lyft","Lime","Spin","Coco"):
            s = t_counts.xs(v, level=0, drop_level=False) if (v in t_counts.index.get_level_values(0)) else pd.Series(dtype="int64")
            aligned = s.reindex(areas.index, fill_value=0)
            w = (aligned + 0.5); w = w / (w.sum() or 1.0)
            mix[v] = w.to_dict()
    else:
        t_counts = trips.groupby(["__vendor__","__area_num__"]).size()
        areas = areas.copy(); areas.set_index("service_area_id", inplace=True)
        mix = {}
        for v in ("Lyft","Lime","Spin","Coco"):
            s = t_counts.xs(v, level=0, drop_level=False) if (v in t_counts.index.get_level_values(0)) else pd.Series(dtype="int64")
            aligned = s.reindex(areas.index, fill_value=0)
            w = (aligned + 0.5); w = w / (w.sum() or 1.0)
            mix[v] = w.to_dict()

    return {"vendor_daily_avg": vendor_daily_avg, "vendor_area_mix": mix}


def size_fleet(vendor_daily_avg: Dict[str,float]) -> Dict[str,int]:
    out: Dict[str,int] = {}
    for v, daily in vendor_daily_avg.items():
        target = TPV_TARGET.get(v, 7.0)
        n = int(math.ceil(daily / max(1e-6, target)))
        n = max(MIN_FLEET_PER_VENDOR if daily > 0 else 0, n)
        out[v] = n
    return out


def generate_vehicles(trips: pd.DataFrame, areas: pd.DataFrame, avg_trip_km: float) -> pd.DataFrame:
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    cal = compute_calibration(trips, areas)
    vendor_daily_avg = cal["vendor_daily_avg"]
    mix = cal["vendor_area_mix"]

    fleet = size_fleet(vendor_daily_avg)
    logger.info("Fleet sizing (vehicles): %s", fleet)

    # service area lookup helpers
    if "__nname__" in areas.columns:
        area_lookup_name = areas.set_index("__nname__")["community_area_name"].to_dict()
        id_lookup_by_name = areas.set_index("__nname__")["service_area_id"].to_dict()
        area_type_by_name = areas.set_index("__nname__")["area_type"].to_dict()
        key_mode = "name"
    else:
        area_lookup_id = areas.set_index("service_area_id")["community_area_name"].to_dict()
        area_type_by_id = areas.set_index("service_area_id")["area_type"].to_dict()
        key_mode = "id"

    rows: List[Dict[str,object]] = []

    start_date = trips["__day__"].min()
    end_date = trips["__day__"].max()
    days_range = (end_date - start_date).days + 1

    for v in ("Lyft","Lime","Spin","Coco"):
        n = fleet.get(v, 0)
        if n <= 0: continue

        # per-vendor distribution over areas
        vmix = mix[v]
        keys = list(vmix.keys())
        weights = np.array([vmix[k] for k in keys], dtype=float)
        weights = weights / (weights.sum() or 1.0)

        # vendor params
        fw_major = FIRMWARE_MAJOR.get(v, 1)
        model_port = MODEL_PORTFOLIO.get(v, [("Generic",1.0)])
        batt_port = BATTERY_PORTFOLIO.get(v, [(520,1.0)])
        tpv_target = TPV_TARGET.get(v, 7.0)
        daily_trips = max(1.0, vendor_daily_avg.get(v, tpv_target))

        for i in range(1, n+1):
            vid = f"{vendor_abbrev(v)}{i:05d}"

            # choose home area
            pick_idx = np_rng.choice(len(keys), p=weights)
            pick_key = keys[pick_idx]
            if key_mode == "name":
                home_id = int(id_lookup_by_name[pick_key])
                home_name = str(area_lookup_name[pick_key])
                area_type = str(area_type_by_name.get(pick_key, "residential"))
            else:
                home_id = int(pick_key)
                home_name = str(area_lookup_id[home_id])
                area_type = str(area_type_by_id.get(home_id, "residential"))

            # attributes
            model = choose_weighted(rng, model_port)
            battery_wh = choose_weighted(rng, batt_port)
            battery_swappable = rng.random() < 0.92
            status = choose_weighted(rng, STATUS_WEIGHTS)

            # dates
            onboard_offset = rng.randint(30, 420)
            in_service_since = (start_date - timedelta(days=onboard_offset)).strftime("%Y-%m-%d")
            decommissioned_at = ""
            if status == "decommissioned":
                # decommission within window
                d0 = start_date + timedelta(days=rng.randint(0, max(1, days_range-1)))
                decommissioned_at = d0.strftime("%Y-%m-%d")

            # firmware & modem
            firmware_version = f"v{fw_major}.{rng.randint(1,4)}.{rng.randint(0,9)}"
            iot_modem = rng.choice(IOT_MODEMS)

            # utilization & odometer
            # assume this vehicle aims for vendor target but with +-25% variation
            veh_tpv = max(1.0, np_rng.lognormal(mean=math.log(tpv_target), sigma=0.25))
            # fraction of life active (maintenance downtime ~5-12%)
            active_frac = 1.0 - (0.05 + 0.07*rng.random())
            est_daily_energy_wh = veh_tpv * ENERGY_PER_TRIP_WH.get(v, 33)

            # days in service until end of window (if decommissioned, less)
            end_for_odo = end_date
            if decommissioned_at:
                end_for_odo = datetime.strptime(decommissioned_at, "%Y-%m-%d").date()
            days_in_service = max(1, (end_for_odo - datetime.strptime(in_service_since, "%Y-%m-%d").date()).days)

            odometer_km = int(max(20, veh_tpv * active_frac * days_in_service * avg_trip_km * (0.85 + 0.3*rng.random())))
            battery_cycles = int((veh_tpv * active_frac * days_in_service * ENERGY_PER_TRIP_WH.get(v,33)) / max(1, battery_wh))

            rows.append({
                "vehicle_id": vid,
                "vendor": v,
                "form_factor": "scooter",
                "model": model,
                "battery_wh": int(battery_wh),
                "battery_swappable": battery_swappable,
                "service_area_id_home": home_id,
                "community_area_name_home": home_name,
                "area_type_home": area_type,
                "in_service_since": in_service_since,
                "decommissioned_at": decommissioned_at,
                "status": status,
                "firmware_version": firmware_version,
                "iot_modem": iot_modem,
                "odometer_km": int(odometer_km),
                "est_daily_trips_target": round(veh_tpv, 2),
                "est_daily_energy_wh": int(est_daily_energy_wh),
                "battery_cycles_est": int(battery_cycles),
            })

    df = pd.DataFrame(rows)
    return df

# ---------------- main ----------------

def main():
    rng = np.random.default_rng(SEED)
    here = Path(__file__).resolve().parent

    trips_path = find_existing(here, TRIPS_FILE_CANDIDATES)
    if not trips_path:
        raise SystemExit("Could not find a trips CSV next to this script (e.g., e_scooter_trips.csv)")
    logger.info("Found trips CSV: %s", trips_path)

    trips, mapping, avg_km = load_trips(trips_path)

    svc = load_service_areas(trips_path.parent)
    areas = build_area_catalog(trips, svc)

    vehicles_df = generate_vehicles(trips, areas, avg_km)

    out_path = trips_path.parent / "vehicles.csv"
    vehicles_df.to_csv(out_path, index=False)
    logger.info("Wrote %d vehicles to %s", len(vehicles_df), out_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        raise
