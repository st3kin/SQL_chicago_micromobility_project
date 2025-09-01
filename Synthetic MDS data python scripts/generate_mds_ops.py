# generate_mds_ops.py — Python 3.8+ / one-shot generator with progress logs
# Place this next to your Chicago trips CSV (any name). Run:  python generate_mds_ops.py
# Outputs to ./mds_export:
#   service_areas.csv
#   hourly_demand_by_vendor.csv
#   vehicles.csv
#   incidents.csv
#   status_changes.csv
#   tasks.csv
#   pricing_policies.csv

import os, sys, math, random, re
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd

# ---------------- configuration (tweak if you want, defaults are fine) ----------------
OUTPUT_DIR = Path("./mds_export")
RANDOM_SEED = 7
CITY_DAILY_BASE = 70000          # baseline city trips/day before seasonality & weekends
WEEKEND_BOOST = 1.08
HOUR_NOISE_SD = 0.06             # hour-level random noise
MIN_VENDOR_FLOORS = {            # keep challengers visible; Lyft/Lime still dominate
    "Lyft": 0.30, "Lime": 0.25, "Spin": 0.08, "Coco": 0.05
}
PROGRESS_EVERY_N_DAYS = 1        # print progress every N days (1 = daily)

# Optional “quick mode” for fast test runs (set to False for full run)
QUICK_MODE = False
QUICK_MAX_AREAS = 30             # use only first N areas
QUICK_MAX_DAYS = 45              # last N days in the window

# Map vendor labels from your file to these four names (extend as needed)
VENDOR_MAP: Dict[str, str] = {
    "lyft": "Lyft", "lime": "Lime", "spin": "Spin", "coco": "Coco",
    "bird": "Coco", "superpedestrian": "Spin", "tier": "Coco"
}

rng = random.Random(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------- helpers ----------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def month_key(d: date) -> str: return f"{d.year}-{d.month:02d}"
def hour_band(h: int) -> str:
    if 0 <= h <= 5: return "late_night"
    if 6 <= h <= 9: return "morning"
    if 10 <= h <= 15: return "midday"
    if 16 <= h <= 19: return "evening"
    return "night"
def seasonal_month_multiplier(m: int) -> float:
    return {1:0.60,2:0.65,3:0.72,4:0.85,5:0.95,6:1.05,7:1.10,8:1.12,9:1.05,10:1.00,11:0.75,12:0.55}.get(m, 1.0)
def hour_shape(h: int) -> float: return {"morning":1.20,"midday":1.00,"evening":1.30,"night":0.85,"late_night":0.55}[hour_band(h)]
def vendor_hour_bias(vendor: str, h: int) -> float:
    b = hour_band(h)
    if vendor == "Lyft": return {"morning":1.05,"midday":1.00,"evening":1.08,"night":0.98,"late_night":0.95}[b]
    if vendor == "Lime": return {"morning":1.02,"midday":1.04,"evening":1.05,"night":1.00,"late_night":0.96}[b]
    if vendor == "Spin": return {"morning":0.95,"midday":0.98,"evening":1.05,"night":1.12,"late_night":1.22}[b]
    if vendor == "Coco": return {"morning":0.92,"midday":0.96,"evening":1.08,"night":1.18,"late_night":1.28}[b]
    return 1.0
def vendor_area_bias(vendor: str, area_type: str) -> float:
    m = 1.0
    if vendor == "Spin" and area_type in ("residential","industrial"): m *= 1.10
    if vendor == "Coco" and area_type == "campus": m *= 1.30
    if vendor == "Lime" and area_type == "downtown": m *= 1.08
    return m
def normalize_with_floors(weights: Dict[str, float], floors: Dict[str, float]) -> Dict[str, float]:
    floors = {k: floors.get(k, 0.0) for k in weights}
    total_floor = sum(floors.values())
    s = sum(max(0.0, v) for v in weights.values()) or 1.0
    out = {k: floors[k] + (1 - total_floor) * (max(0.0, weights[k]) / s) for k in weights}
    ss = sum(out.values())
    return {k: out[k] / ss for k in out}

# ---------------- auto-detect your CSV & columns ----------------
def find_csv() -> Optional[Path]:
    here = Path(".")
    for name in ("chicago_trips.csv", "e_scooter_trips.csv"):
        p = here / name
        if p.exists(): return p
    for p in here.glob("*.csv"):
        return p
    return None

def norm(s: str) -> str: return re.sub(r"[^a-z0-9]", "", s.lower())

def detect_columns(cols: List[str]) -> Dict[str, Optional[str]]:
    ncols = {c: norm(c) for c in cols}
    def match(cands: List[str]) -> Optional[str]:
        cands_n = [norm(c) for c in cands]
        # exact then substring
        for c, nc in ncols.items():
            if nc in cands_n: return c
        for c, nc in ncols.items():
            if any(nc.find(x) >= 0 for x in cands_n): return c
        return None

    start_col = match(["start_time","started_at","trip_start","start_datetime","starttimestamp","startdate","starttime"])
    vendor_col = match(["vendor","provider","provider_name","company","operator"])
    sa_name = match(["start_community_area_name","start_area_name","start_neighborhood","start_zone_name","start_community_area"])
    ea_name = match(["end_community_area_name","end_area_name","end_neighborhood","end_zone_name","end_community_area"])
    sa_num  = match(["start_community_area_number","start_area_id","start_area_code"])
    ea_num  = match(["end_community_area_number","end_area_id","end_area_code"])

    return {"start_time": start_col, "vendor": vendor_col,
            "start_area_name": sa_name, "end_area_name": ea_name,
            "start_area_num": sa_num,  "end_area_num": ea_num}

def load_chicago_metadata(path: Path) -> Dict[str, object]:
    head = pd.read_csv(path, nrows=0)
    mapping = detect_columns(list(head.columns))
    missing_core = [k for k in ("start_time","vendor") if mapping[k] is None]
    if missing_core:
        raise ValueError(f"Could not find required columns for time/vendor. Detected: {mapping}")

    usecols = [c for c in (mapping["start_time"], mapping["vendor"],
                           mapping["start_area_name"], mapping["end_area_name"],
                           mapping["start_area_num"], mapping["end_area_num"]) if c is not None]
    df = pd.read_csv(path, usecols=usecols)

    # Parse start time
    pd.options.mode.use_inf_as_na = True
    df[mapping["start_time"]] = pd.to_datetime(df[mapping["start_time"]], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=[mapping["start_time"]])
    start_date = df[mapping["start_time"]].dt.date.min()
    end_date   = df[mapping["start_time"]].dt.date.max()

    # Canonicalize vendor names
    def canon_vendor(x) -> str:
        key = ("" if pd.isna(x) else str(x)).strip().lower()
        return VENDOR_MAP.get(key, "Coco")
    df["__vendor__"] = df[mapping["vendor"]].apply(canon_vendor)

    # Area names (prefer names; else synthesize from numbers)
    area_names: List[str] = []
    if mapping["start_area_name"] or mapping["end_area_name"]:
        cols = [c for c in (mapping["start_area_name"], mapping["end_area_name"]) if c]
        areas = pd.Series(pd.concat([df[c] for c in cols], ignore_index=True))
        areas = areas.dropna().astype(str).str.strip()
        areas = areas[areas != ""].unique().tolist()
        area_names = areas
    else:
        numcols = [c for c in (mapping["start_area_num"], mapping["end_area_num"]) if c]
        if numcols:
            nums = pd.Series(pd.concat([df[c] for c in numcols], ignore_index=True))
            nums = pd.to_numeric(nums, errors="coerce").dropna().astype(int).unique().tolist()
            area_names = [f"CA {n:02d}" for n in sorted(nums)]
        else:
            area_names = [f"Area {i:02d}" for i in range(1, 25)]

    # Monthly vendor mix (gentle calibration)
    df["__month__"] = df[mapping["start_time"]].dt.strftime("%Y-%m")
    month_mix: Dict[str, Dict[str, float]] = {}
    for mk, sub in df.groupby("__month__"):
        s = sub["__vendor__"].value_counts(normalize=True)
        month_mix[mk] = {v: float(s.get(v, 0.0)) for v in ("Lyft","Lime","Spin","Coco")}

    print("Detected columns:")
    for k, v in mapping.items(): print(f"  {k:16s} -> {v}")
    print(f"Detected vendors (from file): {sorted(set(df['__vendor__']))}")
    print(f"Detected date range: {start_date} .. {end_date}")
    print(f"Service areas (sample): {area_names[:6]}{' ...' if len(area_names) > 6 else ''}")
    return {"start_date": start_date, "end_date": end_date, "areas": area_names, "month_mix": month_mix}

# ---------------- generators ----------------
def build_service_areas(area_names: List[str]) -> pd.DataFrame:
    types = []
    for name in area_names:
        lower = name.lower()
        if any(k in lower for k in ("loop","downtown","central")): t = "downtown"
        elif any(k in lower for k in ("university","campus","college")): t = "campus"
        elif any(k in lower for k in ("industrial","manufacturing","rail yard")): t = "industrial"
        else: t = "residential"
        types.append(t)
    return pd.DataFrame({
        "service_area_id": range(1, len(area_names) + 1),
        "community_area_name": area_names,
        "area_type": types
    })

def generate_hourly(areas_df: pd.DataFrame, vendors: List[str], start: date, end: date,
                    month_mix: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    records = []
    # Area weights
    area_w = {}
    for _, r in areas_df.iterrows():
        base = {"downtown":1.5,"campus":1.0,"industrial":0.8,"residential":1.0}.get(r["area_type"], 1.0)
        base *= rng.uniform(0.7, 1.3)
        area_w[int(r["service_area_id"])] = base
    sum_w = sum(area_w.values())

    total_days = (end - start).days + 1
    cur = start
    day_idx = 0
    print("\n[1/6] Generating hourly_demand_by_vendor …")
    hour_sum = sum(hour_shape(h) for h in range(24))
    while cur <= end:
        day_idx += 1
        if day_idx % PROGRESS_EVERY_N_DAYS == 0:
            print(f"  • Day {day_idx}/{total_days}: {cur}", flush=True)

        mk = month_key(cur)
        day_mult = seasonal_month_multiplier(cur.month)
        if cur.weekday() >= 5: day_mult *= WEEKEND_BOOST
        city_total = int(CITY_DAILY_BASE * day_mult)

        # Monthly baseline + floors + gentle calibration toward real mix
        base = {"Lyft":0.45,"Lime":0.35,"Spin":0.12,"Coco":0.08}
        real = month_mix.get(mk, {})
        w = 0.6
        for v in base: base[v] = (1 - w) * 0.25 + w * real.get(v, 0.0)
        month_share = normalize_with_floors(base, MIN_VENDOR_FLOORS)

        for h in range(24):
            hour_total = city_total * hour_shape(h) / hour_sum
            hour_total *= max(0.5, np.random.normal(1.0, HOUR_NOISE_SD))
            for aid, aw in area_w.items():
                atype = areas_df.loc[areas_df["service_area_id"] == aid, "area_type"].iloc[0]
                band = hour_band(h); mult = 1.0
                if atype == "downtown" and band in ("morning","midday","evening"): mult *= 1.25
                if atype == "downtown" and band in ("night","late_night"): mult *= 0.8
                if atype == "campus" and band in ("night","late_night"): mult *= 1.35
                if atype == "industrial" and band == "morning": mult *= 1.15
                if atype == "industrial" and band in ("night","late_night"): mult *= 0.7
                if atype == "residential" and band in ("night","late_night"): mult *= 1.10
                area_total = hour_total * (aw * mult) / sum_w

                raw = {v: month_share.get(v, 0.0) * vendor_hour_bias(v, h) * vendor_area_bias(v, atype) for v in vendors}
                shares = normalize_with_floors(raw, MIN_VENDOR_FLOORS)
                can = areas_df.loc[areas_df["service_area_id"] == aid, "community_area_name"].iloc[0]
                for v in vendors:
                    lam = max(0.0, area_total * shares[v])
                    trips = int(np.random.poisson(lam))
                    if trips == 0 and lam > 2 and rng.random() < 0.20: trips = 1
                    records.append({
                        "day": cur, "hour": h,
                        "service_area_id": int(aid),
                        "community_area_name": str(can),
                        "vendor": v, "trips": trips
                    })
        cur += timedelta(days=1)

    df = pd.DataFrame.from_records(records)
    g = df.groupby(["day","hour"], as_index=False)["trips"].sum().rename(columns={"trips": "dayhour_total"})
    df = df.merge(g, on=["day","hour"], how="left")
    df["share_within_day"] = df["trips"] / df["dayhour_total"].replace(0, np.nan)
    g2 = df.groupby(["vendor","day"], as_index=False)["trips"].sum().rename(columns={"trips": "vendor_day_total"})
    df = df.merge(g2, on=["vendor","day"], how="left")
    df["share_within_vendor_day"] = df["trips"] / df["vendor_day_total"].replace(0, np.nan)
    print(f"    ↳ hourly_demand_by_vendor rows: {len(df):,}")
    return df[["day","hour","service_area_id","community_area_name","vendor","trips","vendor_day_total","share_within_day","share_within_vendor_day"]]

def estimate_vehicles(df_hourly: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/6] Generating vehicles …", flush=True)
    avg_daily = df_hourly.groupby(["vendor","day"])["trips"].sum().groupby("vendor").mean()
    rows = []
    for vendor, daily in avg_daily.items():
        n = max(60, int(math.ceil(daily / 7.5)))
        for i in range(1, n + 1):
            rows.append({
                "vehicle_id": f"{vendor[:2].upper()}{i:05d}",
                "vendor": vendor,
                "service_area_id_home": int(df_hourly["service_area_id"].sample(1, random_state=rng.randrange(1,10**9)).iloc[0]),
                "community_area_name_home": str(df_hourly["community_area_name"].sample(1, random_state=rng.randrange(1,10**9)).iloc[0]),
                "in_service_since": (pd.Timestamp(df_hourly["day"].min()).date() - timedelta(days=rng.randint(30, 400))).strftime("%Y-%m-%d"),
                "battery_wh": rng.choice([460, 520, 620]),
                "status": rng.choice(["active","active","active","maintenance"])
            })
    df = pd.DataFrame(rows)
    print(f"    ↳ vehicles rows: {len(df):,}")
    return df

def generate_incidents(df_hourly: pd.DataFrame, vehicles: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/6] Generating incidents …", flush=True)
    base_per_1k = {"Lyft":7.5,"Lime":7.0,"Spin":8.0,"Coco":8.5}
    type_mix = {"battery_failure":0.28,"crash":0.22,"illegal_parking":0.30,"vandalism":0.20}
    types = list(type_mix.keys())
    probs = np.array(list(type_mix.values())); probs = probs / probs.sum()
    veh_by_vendor = {v: vehicles.loc[vehicles["vendor"] == v, "vehicle_id"].tolist() for v in vehicles["vendor"].unique()}
    rows = []; inc_id = 1

    days = sorted(df_hourly["day"].unique())
    total_days = len(days)
    for i, d in enumerate(days, 1):
        if i % PROGRESS_EVERY_N_DAYS == 0:
            print(f"  • Day {i}/{total_days}: {d}", flush=True)
        day_df = df_hourly[df_hourly["day"] == d]
        gb = day_df.groupby(["hour","service_area_id","community_area_name","vendor"])
        for (h, aid, can, vendor), sub in gb:
            trips = int(sub["trips"].sum())
            if trips <= 0: continue
            expected = trips * (base_per_1k[vendor] / 1000.0)
            n = int(np.random.poisson(max(0.0, expected)))
            if n == 0: continue
            hour_py = int(h)
            area_id_py = int(aid)
            day_py = pd.Timestamp(d).date()
            chosen = np.random.choice(types, size=n, p=probs, replace=True)
            for t in chosen:
                ts = datetime.combine(day_py, datetime.min.time()) + timedelta(
                    hours=hour_py, minutes=rng.randint(0,59), seconds=rng.randint(0,59))
                resolved = ts + timedelta(minutes=rng.randint(15, 6*60))
                rows.append({
                    "incident_id": inc_id,
                    "vehicle_id": rng.choice(veh_by_vendor.get(vendor, [None])),
                    "vendor": vendor,
                    "occurred_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "incident_type": t,
                    "severity": rng.choices([1,2,3], weights=[0.55,0.35,0.10])[0],
                    "service_area_id": area_id_py,
                    "community_area_name": str(can),
                    "resolved_at": resolved.strftime("%Y-%m-%d %H:%M:%S"),
                    "resolution": rng.choice(["Reset/BMS","Swap battery","On-site fix","No issue found","Rebalance"])
                })
                inc_id += 1
    df = pd.DataFrame(rows)
    print(f"    ↳ incidents rows: {len(df):,}")
    return df

def generate_status_changes(df_hourly: pd.DataFrame, vehicles: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/6] Generating status_changes …", flush=True)
    rows = []; eid = 1
    fleet = vehicles["vehicle_id"].tolist()
    days = sorted(df_hourly["day"].unique())
    total_days = len(days)
    for i, d in enumerate(days, 1):
        if i % PROGRESS_EVERY_N_DAYS == 0:
            print(f"  • Day {i}/{total_days}: {d}", flush=True)
        by_vendor = df_hourly[df_hourly["day"] == d].groupby("vendor")["trips"].sum()
        for vendor, trips in by_vendor.items():
            n_batt = int(max(1, np.random.poisson(trips / 600)))
            n_rebal = int(max(1, np.random.poisson(trips / 750)))
            for _ in range(n_batt):
                ts = datetime.combine(pd.Timestamp(d).date(), datetime.min.time()) + timedelta(
                    hours=rng.randint(7,22), minutes=rng.randint(0,59))
                rows.append({
                    "event_id": eid, "vehicle_id": rng.choice(fleet), "vendor": vendor,
                    "occurred_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_type": "battery_low",
                    "service_area_id": int(df_hourly["service_area_id"].sample(1, random_state=rng.randrange(1,10**9)).iloc[0]),
                    "community_area_name": str(df_hourly["community_area_name"].sample(1, random_state=rng.randrange(1,10**9)).iloc[0]),
                    "meta": "threshold=20%"
                }); eid += 1
            for _ in range(n_rebal):
                ts = datetime.combine(pd.Timestamp(d).date(), datetime.min.time()) + timedelta(
                    hours=rng.randint(5,23), minutes=rng.randint(0,59))
                rows.append({
                    "event_id": eid, "vehicle_id": rng.choice(fleet), "vendor": vendor,
                    "occurred_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_type": "rebalance",
                    "service_area_id": int(df_hourly["service_area_id"].sample(1, random_state=rng.randrange(1,10**9)).iloc[0]),
                    "community_area_name": str(df_hourly["community_area_name"].sample(1, random_state=rng.randrange(1,10**9)).iloc[0]),
                    "meta": rng.choice(["move_to_hotspot","clear_footpath","event_coverage"])
                }); eid += 1
    df = pd.DataFrame(rows)
    print(f"    ↳ status_changes rows: {len(df):,}")
    return df

def generate_tasks(df_hourly: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/6] Generating tasks …", flush=True)
    rows = []; tid = 1
    days = sorted(df_hourly["day"].unique()); total_days = len(days)
    for i, d in enumerate(days, 1):
        if i % PROGRESS_EVERY_N_DAYS == 0:
            print(f"  • Day {i}/{total_days}: {d}", flush=True)
        g = df_hourly[df_hourly["day"] == d]
        day_trips = int(g["trips"].sum())
        n_tasks = int(max(2, np.random.poisson(day_trips / 1500)))
        for _ in range(n_tasks):
            rows.append({
                "task_id": tid,
                "vendor": rng.choice(["Lyft","Lime","Spin","Coco"]),
                "created_at": (datetime.combine(pd.Timestamp(d).date(), datetime.min.time()) + timedelta(
                    hours=rng.randint(6,22))).strftime("%Y-%m-%d %H:%M:%S"),
                "service_area_id": int(g["service_area_id"].sample(1, random_state=rng.randrange(1,10**9)).iloc[0]),
                "community_area_name": str(g["community_area_name"].sample(1, random_state=rng.randrange(1,10**9)).iloc[0]),
                "task_type": rng.choice(["rebalance","battery_swap","repair","pickup_illegal_parking"]),
                "status": rng.choice(["open","assigned","done"]),
                "assigned_to": rng.choice(["tech_01","tech_02","tech_03","tech_04"])
            }); tid += 1
    df = pd.DataFrame(rows)
    print(f"    ↳ tasks rows: {len(df):,}")
    return df

def generate_pricing(vendors: List[str], start: date, end: date) -> pd.DataFrame:
    print("\n[6/6] Generating pricing_policies …", flush=True)
    rows = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        month_end = date(cur.year, 12, 31) if cur.month == 12 else date(cur.year, cur.month + 1, 1) - timedelta(days=1)
        for v in vendors:
            base = {"Lyft":1.15,"Lime":1.10,"Spin":1.05,"Coco":0.99}[v] * rng.uniform(0.95, 1.05)
            per_min = {"Lyft":0.25,"Lime":0.24,"Spin":0.23,"Coco":0.22}[v] * rng.uniform(0.95, 1.05)
            per_km = {"Lyft":0.45,"Lime":0.43,"Spin":0.41,"Coco":0.40}[v] * rng.uniform(0.95, 1.05)
            rows.append({
                "vendor": v,
                "start_date": cur.strftime("%Y-%m-%d"),
                "end_date": month_end.strftime("%Y-%m-%d"),
                "base_fare": round(base, 2),
                "per_minute": round(per_min, 2),
                "per_km": round(per_km, 2)
            })
        cur = month_end + timedelta(days=1)
    df = pd.DataFrame(rows)
    print(f"    ↳ pricing_policies rows: {len(df):,}")
    return df

# ---------------- main ----------------
def main():
    ensure_dir(OUTPUT_DIR)

    # Locate trips CSV
    csv_path = find_csv()
    if not csv_path:
        print("No CSV found. Put your Chicago trips CSV next to this script and run again.")
        sys.exit(1)

    # Inspect CSV + derive metadata
    meta = load_chicago_metadata(csv_path)
    start: date = meta["start_date"]; end: date = meta["end_date"]
    area_names: List[str] = meta["areas"] if meta["areas"] else [f"Area {i:02d}" for i in range(1, 25)]

    # Optional quick mode (for testing smaller outputs)
    if QUICK_MODE:
        end = end
        start = max(start, end - timedelta(days=QUICK_MAX_DAYS - 1))
        area_names = area_names[:QUICK_MAX_AREAS]
        print(f"\n[Quick mode ON] Using {len(area_names)} areas and last {(end - start).days + 1} days.")

    # 1) service_areas
    areas_df = build_service_areas(area_names)
    areas_df.to_csv(OUTPUT_DIR / "service_areas.csv", index=False)
    print(f"\n▶ Wrote service_areas.csv ({len(areas_df):,} rows) to {OUTPUT_DIR.resolve()}")

    vendors = ["Lyft","Lime","Spin","Coco"]

    # 2) hourly_demand_by_vendor
    hourly_df = generate_hourly(areas_df, vendors, start, end, meta["month_mix"])
    hourly_df.to_csv(OUTPUT_DIR / "hourly_demand_by_vendor.csv", index=False)
    print(f"▶ Wrote hourly_demand_by_vendor.csv ({len(hourly_df):,} rows)")

    # 3) vehicles
    vehicles_df = estimate_vehicles(hourly_df)
    vehicles_df.to_csv(OUTPUT_DIR / "vehicles.csv", index=False)
    print(f"▶ Wrote vehicles.csv ({len(vehicles_df):,} rows)")

    # 4) incidents
    incidents_df = generate_incidents(hourly_df, vehicles_df)
    incidents_df.to_csv(OUTPUT_DIR / "incidents.csv", index=False)
    print(f"▶ Wrote incidents.csv ({len(incidents_df):,} rows)")

    # 5) status_changes
    status_df = generate_status_changes(hourly_df, vehicles_df)
    status_df.to_csv(OUTPUT_DIR / "status_changes.csv", index=False)
    print(f"▶ Wrote status_changes.csv ({len(status_df):,} rows)")

    # 6) tasks
    tasks_df = generate_tasks(hourly_df)
    tasks_df.to_csv(OUTPUT_DIR / "tasks.csv", index=False)
    print(f"▶ Wrote tasks.csv ({len(tasks_df):,} rows)")

    # 7) pricing_policies
    pricing_df = generate_pricing(vendors, start, end)
    pricing_df.to_csv(OUTPUT_DIR / "pricing_policies.csv", index=False)
    print(f"▶ Wrote pricing_policies.csv ({len(pricing_df):,} rows)")

    print("\n✅ All files written to:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Partial files remain in", OUTPUT_DIR.resolve())
        sys.exit(1)
